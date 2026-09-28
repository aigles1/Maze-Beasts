// ENet-backed implementation of net.h. This is the only file that sees enet.h / winsock.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "net.h"

#include <enet/enet.h>
#include <ws2tcpip.h>
#include <algorithm>
#include <cstdint>

namespace net {

namespace {
constexpr enet_uint8 CHANNEL_RELIABLE = 0;
constexpr enet_uint8 CHANNEL_UNRELIABLE = 1;
constexpr size_t CHANNEL_COUNT = 2;

// A dropped connection is noticed within 5-10 s instead of ENet's default of up to 30.
void set_timeouts(ENetPeer* p) { enet_peer_timeout(p, 0, 5000, 10000); }
} // namespace

struct Session::Impl {
    bool enet_ready = false;
    ENetHost* host = nullptr;
    bool hosting = false;
    ENetPeer* server = nullptr;      // client side: our connection to the host
    std::vector<ENetPeer*> clients;  // host side: index = peer id, nullptr = free slot

    // Peer ids are stored in ENet's per-peer user pointer, offset by one so that nullptr
    // still means "no id assigned yet".
    static int id_of(ENetPeer* p) { return static_cast<int>(reinterpret_cast<intptr_t>(p->data)) - 1; }

    // ENet reference-counts packets. A packet that no peer accepted (for example, because it
    // disconnected this frame) is still ours to free.
    void send_packet(ENetPeer* p, const std::vector<uint8_t>& msg, bool reliable) {
        ENetPacket* pkt = enet_packet_create(msg.data(), msg.size(), reliable ? ENET_PACKET_FLAG_RELIABLE : 0);
        if (!pkt) return;
        if (!p || enet_peer_send(p, reliable ? CHANNEL_RELIABLE : CHANNEL_UNRELIABLE, pkt) < 0)
            if (pkt->referenceCount == 0) enet_packet_destroy(pkt);
    }
};

Session::Session() : impl(std::make_unique<Impl>()) {
    impl->enet_ready = enet_initialize() == 0;
}

Session::~Session() {
    stop();
    if (impl->enet_ready) enet_deinitialize();
}

bool Session::start_host(uint16_t port, int max_clients, std::string& error) {
    stop();
    if (!impl->enet_ready) { error = "Networking failed to start."; return false; }
    ENetAddress addr;
    addr.host = ENET_HOST_ANY;
    addr.port = port;
    impl->host = enet_host_create(&addr, static_cast<size_t>(max_clients), CHANNEL_COUNT, 0, 0);
    if (!impl->host) {
        error = "Could not open UDP port " + std::to_string(port) + ". Is another copy already hosting?";
        return false;
    }
    impl->hosting = true;
    impl->clients.assign(static_cast<size_t>(max_clients), nullptr);
    return true;
}

bool Session::start_client(const std::string& address, uint16_t port, std::string& error) {
    stop();
    if (!impl->enet_ready) { error = "Networking failed to start."; return false; }
    impl->host = enet_host_create(nullptr, 1, CHANNEL_COUNT, 0, 0);
    if (!impl->host) { error = "Could not create a network socket."; return false; }
    ENetAddress addr;
    if (enet_address_set_host(&addr, address.c_str()) != 0) {
        error = "\"" + address + "\" is not a valid address.";
        stop();
        return false;
    }
    addr.port = port;
    impl->server = enet_host_connect(impl->host, &addr, CHANNEL_COUNT, 0);
    if (!impl->server) { error = "Could not start connecting."; stop(); return false; }
    set_timeouts(impl->server);
    impl->hosting = false;
    return true;
}

void Session::stop() {
    if (!impl->host) return;
    // Notify peers immediately, so they see us leave now rather than after a timeout.
    if (impl->hosting) {
        for (ENetPeer* p : impl->clients) if (p) enet_peer_disconnect_now(p, 0);
    }
    else if (impl->server) {
        enet_peer_disconnect_now(impl->server, 0);
    }
    enet_host_flush(impl->host);
    enet_host_destroy(impl->host);
    impl->host = nullptr;
    impl->server = nullptr;
    impl->clients.clear();
    impl->hosting = false;
}

bool Session::active() const { return impl->host != nullptr; }
bool Session::is_host() const { return impl->host != nullptr && impl->hosting; }

std::vector<Event> Session::poll() {
    std::vector<Event> out;
    if (!impl->host) return out;
    ENetEvent ev;
    while (impl->host && enet_host_service(impl->host, &ev, 0) > 0) {
        switch (ev.type) {
        case ENET_EVENT_TYPE_CONNECT:
            if (impl->hosting) {
                auto slot = std::find(impl->clients.begin(), impl->clients.end(), nullptr);
                if (slot == impl->clients.end()) { enet_peer_disconnect_now(ev.peer, 0); break; }
                int id = static_cast<int>(slot - impl->clients.begin());
                *slot = ev.peer;
                ev.peer->data = reinterpret_cast<void*>(static_cast<intptr_t>(id + 1));
                set_timeouts(ev.peer);
                out.push_back({ Event::Type::Connected, id, {} });
            }
            else {
                out.push_back({ Event::Type::Connected, 0, {} });
            }
            break;
        case ENET_EVENT_TYPE_RECEIVE: {
            int id = impl->hosting ? Impl::id_of(ev.peer) : 0;
            if (id >= 0)
                out.push_back({ Event::Type::Received, id,
                                std::vector<uint8_t>(ev.packet->data, ev.packet->data + ev.packet->dataLength) });
            enet_packet_destroy(ev.packet);
            break;
        }
        case ENET_EVENT_TYPE_DISCONNECT:
            if (impl->hosting) {
                int id = Impl::id_of(ev.peer);
                if (id >= 0 && id < static_cast<int>(impl->clients.size())) {
                    impl->clients[id] = nullptr;
                    ev.peer->data = nullptr;
                    out.push_back({ Event::Type::Disconnected, id, {} });
                }
            }
            else {
                // Also what a failed connection attempt looks like: no Connected came first.
                impl->server = nullptr;
                out.push_back({ Event::Type::Disconnected, 0, {} });
            }
            break;
        default:
            break;
        }
    }
    return out;
}

void Session::send(int peer, const std::vector<uint8_t>& msg, bool reliable) {
    if (!impl->hosting || peer < 0 || peer >= static_cast<int>(impl->clients.size())) return;
    impl->send_packet(impl->clients[peer], msg, reliable);
}

void Session::send_to_host(const std::vector<uint8_t>& msg, bool reliable) {
    if (impl->hosting || !impl->server) return;
    impl->send_packet(impl->server, msg, reliable);
}

void Session::broadcast(const std::vector<uint8_t>& msg, bool reliable, int except_peer) {
    if (!impl->hosting) return;
    for (size_t i = 0; i < impl->clients.size(); i++)
        if (impl->clients[i] && static_cast<int>(i) != except_peer)
            impl->send_packet(impl->clients[i], msg, reliable);
}

void Session::drop(int peer) {
    if (!impl->hosting || peer < 0 || peer >= static_cast<int>(impl->clients.size())) return;
    // A graceful disconnect: queued messages (such as the rejection reason) are delivered first.
    if (impl->clients[peer]) enet_peer_disconnect_later(impl->clients[peer], 0);
}

void Session::flush() {
    if (impl->host) enet_host_flush(impl->host);
}

std::vector<std::string> Session::local_ipv4_addresses() const {
    std::vector<std::string> out;
    if (!impl->enet_ready) return out; // enet_initialize is what started Winsock
    char name[256] = {};
    if (gethostname(name, sizeof(name) - 1) != 0) return out;
    addrinfo hints = {};
    hints.ai_family = AF_INET;
    addrinfo* res = nullptr;
    if (getaddrinfo(name, nullptr, &hints, &res) != 0) return out;
    for (addrinfo* p = res; p; p = p->ai_next) {
        char buf[INET_ADDRSTRLEN] = {};
        auto* sin = reinterpret_cast<sockaddr_in*>(p->ai_addr);
        if (!inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf))) continue;
        std::string ip = buf;
        if (ip.rfind("127.", 0) == 0 || ip.rfind("169.254.", 0) == 0) continue;
        if (std::find(out.begin(), out.end(), ip) == out.end()) out.push_back(ip);
    }
    freeaddrinfo(res);
    return out;
}

} // namespace net
