// The maze world, shared by the game and the dedicated server: building a maze, its lower
// level, monsters and medpacks from a seed, and the monsters' AI. Nothing here touches
// graphics, sound or the network, so it compiles on its own for a headless Linux server.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

struct Monster {
    double x, y;
    int hp;
    int type;          // 1 = monster, 2 = boss
    double target_x, target_y;
    int cooldown;      // ticks until it may fire again
    int hit_flash = 0; // ticks remaining to render this monster tinted red after being hit
    int id = 0;        // stable across the network: assigned in spawn order, identical everywhere
    double net_x = 0.0, net_y = 0.0; // latest networked position, which x/y glide toward
};

struct HealthPack {
    double x, y;
    int level = 0; // 0 = maze, 1 = lower level; both share the same x/y footprint
    int id = 0;    // stable across the network, so a pickup names the right pack
};

struct Projectile {
    double x, y;
    double dir_x, dir_y;
    double speed;
    double z = 0.5;      // world height (upper level spans 0..1, lower level spans -2..0)
    double dir_z = 0.0;  // vertical component of travel direction
    bool from_boss = false; // boss shots hit harder and render smaller
    int level = 0;       // which level's walls this shot collides with (0 = maze, 1 = lower)
    int owner = 0;       // who fired it: 0 = a monster, 1-3 = that player's slot
};

// Random numbers that come out the same on every platform. std::mt19937 itself is fully
// specified by the C++ standard, but its helpers are not: uniform_int_distribution, shuffle and
// friends turn the same stream into different numbers under MSVC and GCC. The Windows game and
// the Linux server build every maze from a shared seed, so world generation uses these instead.
class PortableRng {
public:
    void seed(uint32_t s) { engine.seed(s); }
    uint32_t next() { return static_cast<uint32_t>(engine()); }

    // lo..hi inclusive. The modulo's tiny bias doesn't matter for a maze.
    int range(int lo, int hi) {
        return lo + static_cast<int>(next() % (static_cast<uint32_t>(hi - lo) + 1u));
    }

    // lo..hi, excluding hi.
    double real(double lo, double hi) { return lo + (hi - lo) * (next() * (1.0 / 4294967296.0)); }

    // True with probability p.
    bool chance(double p) { return next() < static_cast<uint32_t>(p * 4294967296.0); }

    // An index picked with probability proportional to its (whole-number) weight.
    size_t weighted(const std::vector<uint32_t>& weights) {
        uint64_t total = 0;
        for (uint32_t w : weights) total += w;
        uint64_t r = total ? next() % total : 0;
        for (size_t i = 0; i < weights.size(); i++) {
            if (r < weights[i]) return i;
            r -= weights[i];
        }
        return weights.empty() ? 0 : weights.size() - 1;
    }

    // Fisher-Yates.
    template <typename T> void shuffle(std::vector<T>& v) {
        for (size_t i = v.size(); i > 1; i--) {
            size_t j = static_cast<size_t>(range(0, static_cast<int>(i) - 1));
            std::swap(v[i - 1], v[j]);
        }
    }

private:
    std::mt19937 engine;
};

class World {
public:
    using Cell = std::pair<int, int>;
    using Connections = std::map<Cell, std::set<Cell>>;

    // Lower level: a wide-open floor beneath the maze, reached by a staircase near the start.
    // It shares the maze's grid footprint, so (x, y) coordinates carry across both levels.
    static constexpr double lower_floor_y = -2.0; // world height of the lower floor

    int grid_size = 33;
    std::vector<std::vector<int>> grid;       // grid[y][x]: 0 = open, 1 = solid
    Connections connections;                  // which neighbouring cells you can walk between
    Cell start, end;
    std::vector<std::tuple<int, int, int>> rooms; // boss rooms: x, y, size
    std::vector<std::vector<int>> lower_grid;
    Connections lower_connections;
    std::vector<Cell> stair_cells;            // the staircase run, ordered top to bottom
    std::vector<Monster> monsters;
    std::vector<HealthPack> health_packs;
    PortableRng rng;

    // A whole world from one seed. Every machine that calls this with the same seed gets the
    // same maze, monsters and medpacks, whatever compiler or operating system built it.
    void generate(uint32_t seed);

    static uint32_t random_seed();

    // Player 2 starts at the exit and escapes through Player 1's start, so the two swap ends.
    Cell spawn_cell(int slot) const { return slot == 2 ? end : start; }
    Cell exit_cell(int slot) const { return slot == 2 ? start : end; }

    // A heading, in degrees, that looks down an open passage from a cell instead of at a wall.
    double open_facing_yaw(Cell c);

    const std::vector<std::vector<int>>& grid_for(int level) const { return level == 0 ? grid : lower_grid; }
    Connections& conn_for(int level) { return level == 0 ? connections : lower_connections; }

    bool is_stair_cell(int x, int y) const;
    // How far along the staircase a position is: 0 at the top step, 1 at the bottom.
    double stair_progress(double x) const;
    // World height of the floor under a position on a level; on the stairs it ramps between.
    double floor_height(double x, double y, int level) const;
    // True if nothing blocks a straight line between two points on a level's walls.
    bool line_of_sight(int level, double x0, double y0, double x1, double y1);

    Monster* find_monster(int id);
    const Monster* find_monster(int id) const;
    int boss_count() const;
    // Authoritative damage. Returns true if that killed it (it is then removed).
    bool damage_monster(int id, int dmg);

    // Someone a monster can shoot at: an explorer standing on the maze level.
    struct Target { double x, y; };
    // One tick of monster AI (the game runs it once a frame, the server 60 times a second).
    // `driven_boss` is the boss a player steers as the Beast, or -1: it doesn't think for
    // itself, it glides toward the position its player last reported (net_x, net_y). Shots the
    // monsters decide to fire are appended to `shots`.
    void step_monsters(double delta, int driven_boss, const std::vector<Target>& targets,
                       std::vector<Projectile>& shots);

    // A fingerprint of the generated world. Everyone logs it when a round starts, so a
    // mismatch - two machines building different mazes from one seed - is easy to spot.
    uint64_t world_checksum() const;

private:
    static void link(Connections& conn, Cell a, Cell b);
    static void unlink(Connections& conn, Cell a, Cell b);
    std::optional<std::vector<Cell>> find_path();
    void generate_maze();
    void generate_lower_level();
    void carve_staircase();
    void spawn_monsters();
    void spawn_health_packs();
};
