// Networking for multiplayer, as a thin layer over ENet (reliable UDP).
//
// This header deliberately does not include enet.h. ENet pulls in <winsock2.h>, and the main
// game source already includes <windows.h> through miniaudio; mixing the two in one translation
// unit causes redefinition errors. Everything ENet-specific lives in net.cpp instead.
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace net {

constexpr uint16_t DEFAULT_PORT = 29180; // UDP. Below Windows' ephemeral range (49152+) and not
                                         // the default of any well-known game or service.

struct Event {
    enum class Type { Connected, Disconnected, Received };
    Type type;
    int peer = -1;              // host: which client (0-based); client: always 0, the host
    std::vector<uint8_t> data;  // Received only
};

// One network endpoint: either the host (a server that clients connect to) or a client.
// Messages go out on two channels: reliable-ordered for events that must arrive (shots, deaths,
// round changes) and unreliable-sequenced for high-rate state where only the newest matters.
class Session {
public:
    Session();
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    bool start_host(uint16_t port, int max_clients, std::string& error);
    bool start_client(const std::string& address, uint16_t port, std::string& error);
    void stop(); // tells every connected peer we're leaving, then frees the socket

    bool active() const;
    bool is_host() const;

    std::vector<Event> poll(); // pump the network; call once per frame

    void send(int peer, const std::vector<uint8_t>& msg, bool reliable);  // host -> one client
    void send_to_host(const std::vector<uint8_t>& msg, bool reliable);    // client -> host
    void broadcast(const std::vector<uint8_t>& msg, bool reliable, int except_peer = -1); // host
    void drop(int peer); // host: disconnect one client, e.g. after rejecting it
    void flush();        // push queued packets out now instead of at the next poll

    // This machine's IPv4 addresses (loopback and link-local excluded), to show the host
    // what to tell the other players.
    std::vector<std::string> local_ipv4_addresses() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Minimal binary message writer/reader. Values are copied in native byte order: every build of
// this game is x86-64 Windows, so both ends always agree.
class Writer {
public:
    template <typename T> Writer& put(T v) {
        size_t n = buf.size();
        buf.resize(n + sizeof(T));
        std::memcpy(buf.data() + n, &v, sizeof(T));
        return *this;
    }
    std::vector<uint8_t> buf;
};

// Reads never run past the end of the message: a short or malformed packet from the network
// just clears ok and yields zeros, so handlers check ok before trusting what they read.
class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& d) : data(d) {}
    template <typename T> T get() {
        T v{};
        if (pos + sizeof(T) <= data.size()) std::memcpy(&v, data.data() + pos, sizeof(T));
        else ok = false;
        pos += sizeof(T);
        return v;
    }
    bool ok = true;
private:
    const std::vector<uint8_t>& data;
    size_t pos = 0;
};

} // namespace net
