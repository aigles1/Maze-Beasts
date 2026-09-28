// MazeBeasts dedicated server.
//
// A headless stand-in for the player who would otherwise host: 2-3 players connect to it
// directly by IP address and port, and it does everything a hosting player's game does except
// play. It builds the same maze as the players from a shared seed (world.cpp), runs the
// monsters and bosses, decides medpack pickups and who escaped first, starts the rounds, and
// relays every player's messages to the others. It has no window, sound or graphics, so it
// builds on Linux (see the Makefile) as well as Windows, and can run on a cloud machine such
// as an AWS EC2 instance.
//
// Slots go out in the order players join: the first is Player 1 (starts where singleplayer
// does, and has the "Start the game" button), the second Player 2 (starts at the exit), the
// third the Beast. In the lobby, and before each new maze, players move up to fill any gap.

#include "net.h"
#include "protocol.h"
#include "world.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <timeapi.h>
#endif

namespace {

const char* const VERSION = "0.41";
constexpr double TICK = 1.0 / 60.0;   // monster AI rate: what a hosting player's game runs at 60 fps
constexpr int MAX_PLAYERS = 3;
constexpr int SPARE_CONNECTIONS = 2;  // so a 4th player is told "full" rather than timing out

volatile std::sig_atomic_t stop_requested = 0;
void on_signal(int) { stop_requested = 1; }

void log_line(const std::string& line) {
    std::time_t t = std::time(nullptr);
    char stamp[16] = "";
    if (const std::tm* tm = std::localtime(&t)) std::strftime(stamp, sizeof(stamp), "%H:%M:%S", tm);
    std::cout << "[" << stamp << "] " << line << std::endl;
}

std::string name(int slot) { return slot == 3 ? "Player 3 (the Beast)" : "Player " + std::to_string(slot); }

class Server {
public:
    Server(uint16_t port, int autostart, uint32_t first_seed)
        : port(port), autostart(autostart), next_seed(first_seed), t0(std::chrono::steady_clock::now()) {}

    bool start(std::string& error) { return net.start_host(port, MAX_PLAYERS + SPARE_CONNECTIONS, error); }

    // Until Ctrl+C (or a service manager's stop). Waits for network traffic between ticks, so
    // messages are relayed the moment they arrive and an idle server uses next to no CPU.
    void run() {
        double next_tick = now();
        while (!stop_requested) {
            double wait = next_tick - now();
            for (const auto& e : net.poll(wait > 0.0 ? static_cast<int>(wait * 1000.0) : 0)) {
                switch (e.type) {
                case net::Event::Type::Connected:    log_line("connection from " + net.peer_address(e.peer)); break;
                case net::Event::Type::Disconnected: on_disconnected(e.peer); break;
                case net::Event::Type::Received:     on_message(e.peer, e.data); break;
                }
            }
            double t = now();
            if (t >= next_tick) {
                tick(t);
                next_tick += TICK;
                if (t - next_tick > 0.25) next_tick = t; // badly behind (the machine stalled): don't race to catch up
            }
            net.flush();
        }
    }

    void shutdown() { net.stop(); } // tells every player we've gone, rather than leaving them to time out

private:
    struct Slot {
        int peer = -1;            // network peer holding this slot, -1 = free
        std::string address;
        bool playing = false;     // in the current round (as opposed to waiting for the next one)
        bool has_state = false;   // has reported a position this round
        double x = 0.0, y = 0.0;
        int level = 0;
        bool alive = true;
    };

    net::Session net;
    World world;
    Slot slots[MAX_PLAYERS + 1]; // [1..3]
    uint16_t port;
    int autostart;               // start by itself once this many have joined (0 = Player 1 decides)
    uint32_t next_seed;          // the next maze's seed (shown to players as they join)
    bool in_round = false, round_over = false;
    uint8_t round_id = 0;
    double round_over_until = 0.0, monster_timer = 0.0;
    int beast_boss_id = -1;      // boss Player 3 steers, or -1
    std::chrono::steady_clock::time_point t0;

    double now() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

    int slot_of(int peer) const {
        for (int s = 1; s <= MAX_PLAYERS; s++) if (slots[s].peer == peer) return s;
        return -1;
    }

    int player_count() const {
        int n = 0;
        for (int s = 1; s <= MAX_PLAYERS; s++) if (slots[s].peer >= 0) n++;
        return n;
    }

    uint8_t lobby_mask() const {
        uint8_t mask = 0;
        for (int s = 1; s <= MAX_PLAYERS; s++) if (slots[s].peer >= 0) mask = static_cast<uint8_t>(mask | (1 << s));
        return mask;
    }

    bool beast_driven() const { return beast_boss_id >= 0 && slots[3].peer >= 0 && slots[3].playing; }

    void send(int peer, const net::Writer& w) { net.send(peer, w.buf, true); }
    void broadcast(const net::Writer& w, bool reliable = true) { net.broadcast(w.buf, reliable); }

    void broadcast_lobby() {
        net::Writer w;
        w.put<uint8_t>(MSG_LOBBY).put<uint8_t>(lobby_mask());
        broadcast(w);
    }

    // Move players down to fill gaps, so slots run 1, 2, 3 with nobody missing - there is
    // always a Player 1 to start the game, and explorers before the Beast.
    void compact() {
        int next = 1;
        for (int s = 1; s <= MAX_PLAYERS; s++) {
            if (slots[s].peer < 0) continue;
            if (s != next) {
                slots[next] = slots[s];
                slots[s] = Slot{};
                net::Writer w;
                w.put<uint8_t>(MSG_ASSIGN).put<uint8_t>(static_cast<uint8_t>(next));
                send(slots[next].peer, w);
                log_line(slots[next].address + " moves up to " + name(next));
            }
            next++;
        }
    }

    // ---- Players arriving and leaving ------------------------------------------------------

    void on_hello(int peer, net::Reader& r) {
        uint8_t version = r.get<uint8_t>();
        if (slot_of(peer) >= 0) return; // already joined
        int free_slot = -1;
        for (int s = 1; s <= MAX_PLAYERS && free_slot < 0; s++) if (slots[s].peer < 0) free_slot = s;
        uint8_t reason = 0;
        if (!r.ok || version != PROTOCOL_VERSION) reason = REJECT_VERSION;
        else if (free_slot < 0) reason = REJECT_FULL;
        if (reason) {
            net::Writer w;
            w.put<uint8_t>(MSG_REJECT).put<uint8_t>(reason);
            send(peer, w);
            net.drop(peer);
            log_line("turned away " + net.peer_address(peer) + (reason == REJECT_FULL ? ": the game is full"
                : ": a different version of MazeBeasts (protocol " + std::to_string(version) + ", this server speaks "
                  + std::to_string(PROTOCOL_VERSION) + ")"));
            return;
        }
        Slot& p = slots[free_slot];
        p = Slot{};
        p.peer = peer;
        p.address = net.peer_address(peer);
        uint8_t flags = static_cast<uint8_t>(WELCOME_DEDICATED | (in_round ? WELCOME_IN_PROGRESS : 0));
        net::Writer w;
        w.put<uint8_t>(MSG_WELCOME).put<uint8_t>(static_cast<uint8_t>(free_slot)).put<uint32_t>(next_seed).put<uint8_t>(flags);
        send(peer, w);
        broadcast_lobby();
        log_line(name(free_slot) + " joined from " + p.address + " (" + std::to_string(player_count()) + "/3)"
            + (in_round ? " - plays from the next maze" : ""));
    }

    void on_disconnected(int peer) {
        int s = slot_of(peer);
        if (s < 0) return; // never finished joining
        log_line(name(s) + " left (" + slots[s].address + ")");
        bool was_playing = slots[s].playing;
        slots[s] = Slot{};

        if (player_count() == 0) {
            if (in_round) log_line("everyone has left - back to waiting for players");
            in_round = false;
            round_over = false;
            beast_boss_id = -1;
            return;
        }
        if (!in_round) {
            compact();
            broadcast_lobby();
            return;
        }
        if (s == 3 && was_playing && beast_boss_id >= 0) {
            // Nobody drives that boss any more: hand it back to its AI.
            beast_boss_id = -1;
            net::Writer w;
            w.put<uint8_t>(MSG_BEAST_BOSS).put<uint8_t>(round_id).put<int32_t>(-1);
            broadcast(w);
        }
        bool explorers = false;
        for (int e = 1; e <= 2; e++) if (slots[e].peer >= 0 && slots[e].playing) explorers = true;
        if (!explorers) {
            back_to_lobby("no explorers left in this maze - back to the lobby");
            return;
        }
        broadcast_lobby();
    }

    // ---- Rounds ------------------------------------------------------------------------------

    void start_round() {
        compact(); // the lowest slots are the explorers
        round_id++;
        uint32_t seed = next_seed;
        next_seed = World::random_seed();
        world.generate(seed);
        uint8_t mask = lobby_mask();
        for (int s = 1; s <= MAX_PLAYERS; s++) {
            Slot& p = slots[s];
            p.playing = p.peer >= 0;
            p.has_state = false;
            p.alive = true;
        }
        beast_boss_id = -1;
        if (slots[3].peer >= 0) {
            std::vector<int> bosses;
            for (const auto& m : world.monsters) if (m.type == 2) bosses.push_back(m.id);
            if (!bosses.empty()) beast_boss_id = bosses[static_cast<size_t>(world.rng.range(0, static_cast<int>(bosses.size()) - 1))];
        }
        net::Writer w;
        w.put<uint8_t>(MSG_START).put<uint8_t>(round_id).put<uint32_t>(seed).put<int32_t>(beast_boss_id).put<uint8_t>(mask);
        broadcast(w);
        in_round = true;
        round_over = false;
        monster_timer = 0.0;
        log_line("maze " + std::to_string(round_id) + " started with " + std::to_string(player_count()) + " player(s): seed "
            + std::to_string(seed) + ", world checksum " + std::to_string(world.world_checksum()) + ", "
            + std::to_string(world.boss_count()) + " bosses" + (beast_boss_id >= 0 ? ", the Beast is boss " + std::to_string(beast_boss_id) : ""));
    }

    void end_round(int winner) {
        round_over = true;
        round_over_until = now() + ROUND_OVER_SECONDS;
        net::Writer w;
        w.put<uint8_t>(MSG_ROUND_OVER).put<uint8_t>(round_id).put<uint8_t>(static_cast<uint8_t>(winner));
        broadcast(w);
        log_line(name(winner) + " escaped the maze - next maze in " + std::to_string(static_cast<int>(ROUND_OVER_SECONDS)) + " s");
    }

    void back_to_lobby(const std::string& why) {
        in_round = false;
        round_over = false;
        beast_boss_id = -1;
        for (int s = 1; s <= MAX_PLAYERS; s++) slots[s].playing = false;
        net::Writer w;
        w.put<uint8_t>(MSG_TO_LOBBY);
        broadcast(w);
        compact();
        broadcast_lobby();
        log_line(why);
    }

    // The Beast's boss died: give Player 3 another living boss, if there is one.
    void reassign_beast() {
        beast_boss_id = -1;
        if (slots[3].peer >= 0 && slots[3].playing) {
            std::vector<int> bosses;
            for (const auto& m : world.monsters) if (m.type == 2) bosses.push_back(m.id);
            if (!bosses.empty()) {
                beast_boss_id = bosses[static_cast<size_t>(world.rng.range(0, static_cast<int>(bosses.size()) - 1))];
                if (Monster* m = world.find_monster(beast_boss_id)) { m->net_x = m->x; m->net_y = m->y; }
            }
        }
        net::Writer w;
        w.put<uint8_t>(MSG_BEAST_BOSS).put<uint8_t>(round_id).put<int32_t>(beast_boss_id);
        broadcast(w);
        log_line(beast_boss_id >= 0 ? "the Beast's boss died - Player 3 now controls boss " + std::to_string(beast_boss_id)
                               : "the Beast's boss died - no bosses left for Player 3");
    }

    void tick(double t) {
        if (!in_round) {
            if (autostart >= 2 && player_count() >= autostart) start_round();
            return;
        }
        // Monsters shoot at explorers standing on the maze level.
        std::vector<World::Target> targets;
        for (int s = 1; s <= 2; s++) {
            const Slot& p = slots[s];
            if (p.peer >= 0 && p.playing && p.has_state && p.alive && p.level == 0) targets.push_back({ p.x, p.y });
        }
        std::vector<Projectile> shots;
        world.step_monsters(TICK, beast_driven() ? beast_boss_id : -1, targets, shots);
        for (const auto& p : shots) {
            net::Writer w;
            w.put<uint8_t>(MSG_SHOT).put<uint8_t>(round_id).put<uint8_t>(0).put<uint8_t>(0).put<uint8_t>(p.from_boss ? 1 : 0)
             .put<float>(static_cast<float>(p.x)).put<float>(static_cast<float>(p.y)).put<float>(static_cast<float>(p.z))
             .put<float>(static_cast<float>(p.dir_x)).put<float>(static_cast<float>(p.dir_y))
             .put<float>(static_cast<float>(p.dir_z)).put<float>(static_cast<float>(p.speed));
            broadcast(w);
        }
        monster_timer -= TICK;
        if (monster_timer <= 0.0) {
            monster_timer = MONSTER_INTERVAL;
            send_monsters();
        }
        if (round_over && t >= round_over_until) start_round();
    }

    // Where every monster is, 20 times a second. Unreliable on purpose: each snapshot is
    // complete, so a lost one is simply replaced by the next.
    void send_monsters() {
        size_t n = std::min<size_t>(world.monsters.size(), 512);
        net::Writer w;
        w.put<uint8_t>(MSG_MONSTERS).put<uint8_t>(round_id).put<uint16_t>(static_cast<uint16_t>(n));
        for (size_t i = 0; i < n; i++) {
            const Monster& m = world.monsters[i];
            w.put<int32_t>(m.id).put<float>(static_cast<float>(m.x)).put<float>(static_cast<float>(m.y))
             .put<int32_t>(m.hp).put<uint8_t>(static_cast<uint8_t>(m.type))
             .put<uint8_t>(static_cast<uint8_t>(std::clamp(m.hit_flash, 0, 255)));
        }
        broadcast(w, false);
    }

    // ---- Messages ------------------------------------------------------------------------------

    void on_message(int peer, const std::vector<uint8_t>& data) {
        net::Reader r(data);
        uint8_t type = r.get<uint8_t>();
        if (type == MSG_HELLO) { on_hello(peer, r); return; }
        int s = slot_of(peer);
        if (s < 0) return; // hasn't said hello yet

        if (type == MSG_CHAT) {
            // Passed on to everyone else, labelled with who said it. Chat isn't tied to a
            // round, so it works in the lobby too.
            std::string text = trim_spaces(clean_chat_text(r.get_text()));
            if (!r.ok || text.empty()) return;
            net::Writer w;
            w.put<uint8_t>(MSG_CHAT).put<uint8_t>(static_cast<uint8_t>(s)).put_text(text);
            net.broadcast(w.buf, true, peer);
            log_line("[chat] " + name(s) + ": " + text);
            return;
        }

        if (type == MSG_REQUEST_START) {
            // Player 1's Start button in the lobby, or F8 for a fresh maze during a game.
            if (s != 1) return;
            if (!in_round && player_count() >= 2) start_round();
            else if (in_round && slots[1].playing) {
                log_line("Player 1 asked for a new maze");
                start_round();
            }
            return;
        }

        uint8_t rnd = r.get<uint8_t>();
        if (!r.ok || !in_round || rnd != round_id || !slots[s].playing) return; // an old maze, or not in this one
        switch (type) {
        case MSG_STATE:
            if (read_state(r, s)) net.broadcast(data, false, peer);
            break;
        case MSG_SHOT: {
            int owner = r.get<uint8_t>();
            if (r.ok && owner == s) net.broadcast(data, true, peer);
            break;
        }
        case MSG_MONSTER_HIT: {
            int id = r.get<int32_t>();
            int dmg = r.get<int32_t>();
            if (!r.ok || dmg <= 0 || dmg > 300) break;
            bool was_beast = id == beast_boss_id;
            bool boss = world.find_monster(id) && world.find_monster(id)->type == 2;
            if (world.damage_monster(id, dmg)) {
                if (boss) log_line(name(s) + " killed a boss (" + std::to_string(world.boss_count()) + " left)");
                if (was_beast) reassign_beast();
            }
            break;
        }
        case MSG_DEATH: {
            int victim = r.get<uint8_t>();
            int killer = r.get<uint8_t>();
            if (!r.ok || victim != s || killer > 3) break;
            net.broadcast(data, true, peer);
            slots[s].alive = false;
            log_line(name(victim) + " was killed by " + (killer == 0 ? std::string("a monster") : name(killer)));
            break;
        }
        case MSG_PICKUP: {
            int id = r.get<int32_t>();
            if (!r.ok) break;
            auto& packs = world.health_packs;
            auto it = std::find_if(packs.begin(), packs.end(), [id](const HealthPack& h) { return h.id == id; });
            if (it == packs.end()) break; // someone got there first
            packs.erase(it);
            net::Writer w;
            w.put<uint8_t>(MSG_PACK_GONE).put<uint8_t>(round_id).put<int32_t>(id);
            broadcast(w);
            break;
        }
        case MSG_REACHED_EXIT: {
            int who = r.get<uint8_t>();
            // The server has the final say on whether every boss is really dead.
            if (r.ok && who == s && s <= 2 && !round_over && world.boss_count() == 0) end_round(who);
            break;
        }
        default:
            break;
        }
    }

    // A player's position report. Only their own: nobody can move someone else.
    bool read_state(net::Reader& r, int sender) {
        int slot = r.get<uint8_t>();
        int level = r.get<uint8_t>();
        uint8_t flags = r.get<uint8_t>();
        float x = r.get<float>(), y = r.get<float>(), jump = r.get<float>();
        float yaw = r.get<float>(), pitch = r.get<float>();
        r.get<int16_t>(); // hp: the players show it; the server doesn't need it
        if (!r.ok || slot != sender) return false;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(jump) || !std::isfinite(yaw) || !std::isfinite(pitch)) return false;
        Slot& p = slots[slot];
        double nx = std::clamp(static_cast<double>(x), 0.0, world.grid_size - 0.001);
        double ny = std::clamp(static_cast<double>(y), 0.0, world.grid_size - 0.001);
        bool teleport = !p.has_state || std::hypot(nx - p.x, ny - p.y) > 2.0;
        p.x = nx;
        p.y = ny;
        p.level = level != 0 ? 1 : 0;
        p.alive = (flags & 1) != 0;
        if (!p.has_state) log_line(name(slot) + " is in the maze");
        p.has_state = true;
        // The Beast's boss follows Player 3.
        if (slot == 3 && beast_boss_id >= 0) {
            if (Monster* m = world.find_monster(beast_boss_id)) {
                m->net_x = nx;
                m->net_y = ny;
                if (teleport) { m->x = nx; m->y = ny; }
            }
        }
        return true;
    }
};

void usage() {
    std::cout <<
        "MazeBeasts dedicated server " << VERSION << "\n"
        "\n"
        "Usage: mazebeasts-server [--port N] [--autostart N] [--seed N]\n"
        "\n"
        "  --port N       UDP port to listen on (default " << net::DEFAULT_PORT << ")\n"
        "  --autostart N  start the first maze by itself once N players (2 or 3) have joined;\n"
        "                 otherwise Player 1 starts it from their lobby screen\n"
        "  --seed N       the first maze's seed (later mazes are random)\n"
        "\n"
        "Players connect with Multiplayer - Join, typing this machine's IP address (and\n"
        ":port if it isn't " << net::DEFAULT_PORT << "). Stop the server with Ctrl+C.\n";
}

} // namespace

int main(int argc, char** argv) {
    uint16_t port = net::DEFAULT_PORT;
    int autostart = 0;
    uint32_t seed = World::random_seed();
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto value = [&](const std::string& key, std::string& out) {
            // Accepts both "--key=value" and "--key value".
            if (a.rfind(key + "=", 0) == 0) { out = a.substr(key.size() + 1); return true; }
            if (a == key && i + 1 < argc) { out = argv[++i]; return true; }
            return false;
        };
        std::string v;
        if (a == "--help" || a == "-h" || a == "/?") { usage(); return 0; }
        else if (value("--port", v)) {
            long p = std::atol(v.c_str());
            if (p < 1 || p > 65535) { std::cerr << "--port must be 1-65535\n"; return 2; }
            port = static_cast<uint16_t>(p);
        }
        else if (value("--autostart", v)) autostart = std::atoi(v.c_str());
        else if (value("--seed", v)) seed = static_cast<uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
        else { std::cerr << "Unknown option: " << a << "\n\n"; usage(); return 2; }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#ifdef SIGBREAK
    std::signal(SIGBREAK, on_signal); // Ctrl+Break, or closing the console window, on Windows
#endif
#ifdef _WIN32
    timeBeginPeriod(1); // Windows otherwise wakes a waiting thread only every ~16 ms
#endif

    Server server(port, autostart, seed);
    std::string err;
    if (!server.start(err)) {
        std::cerr << "Could not open UDP port " << port << ": is another server already running on it?\n";
        return 1;
    }
    log_line(std::string("MazeBeasts dedicated server ") + VERSION + " (protocol " + std::to_string(PROTOCOL_VERSION)
        + ") listening on UDP port " + std::to_string(port));
    log_line(autostart >= 2 ? "the first maze starts once " + std::to_string(autostart) + " players have joined"
                       : std::string("waiting for players - Player 1 starts the game from their lobby"));
    server.run();
    log_line("shutting down");
    server.shutdown();
#ifdef _WIN32
    timeEndPeriod(1);
#endif
    return 0;
}
