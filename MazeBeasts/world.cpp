#include "world.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <stack>

void World::link(Connections& conn, Cell a, Cell b) {
    conn[a].insert(b);
    conn[b].insert(a);
}

void World::unlink(Connections& conn, Cell a, Cell b) {
    conn[a].erase(b);
    conn[b].erase(a);
}

uint32_t World::random_seed() {
    std::random_device rd;
    return rd() ^ static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
}

// Every random draw below comes from rng in a fixed order, so one seed always rebuilds the
// same world.
void World::generate(uint32_t seed) {
    rng.seed(seed);
    generate_maze();
    generate_lower_level();
    carve_staircase();
    spawn_monsters();
    spawn_health_packs();
}

double World::open_facing_yaw(Cell c) {
    const int dxs[4] = { 1, 0, -1, 0 };
    const int dys[4] = { 0, 1, 0, -1 };
    for (int i = 0; i < 4; i++)
        if (connections[c].count({ c.first + dxs[i], c.second + dys[i] }))
            return std::atan2(static_cast<double>(dys[i]), static_cast<double>(dxs[i])) * 180.0 / 3.14159265358979323846;
    return 0.0;
}

bool World::is_stair_cell(int x, int y) const {
    for (const auto& c : stair_cells) if (c.first == x && c.second == y) return true;
    return false;
}

// The run is laid out along +x, so only the x coordinate matters.
double World::stair_progress(double x) const {
    if (stair_cells.empty()) return 0.0;
    double t = (x - stair_cells.front().first) / static_cast<double>(stair_cells.size());
    return std::clamp(t, 0.0, 1.0);
}

double World::floor_height(double x, double y, int level) const {
    if (is_stair_cell(static_cast<int>(x), static_cast<int>(y)))
        return stair_progress(x) * lower_floor_y;
    return level == 0 ? 0.0 : lower_floor_y;
}

bool World::line_of_sight(int level, double x0, double y0, double x1, double y1) {
    const auto& g = grid_for(level);
    auto& conn = conn_for(level);
    double dx = x1 - x0, dy = y1 - y0;
    double dist = std::hypot(dx, dy);
    int steps = std::max(1, static_cast<int>(std::ceil(dist / 0.05)));
    double sx = dx / steps, sy = dy / steps;
    double cx = x0, cy = y0;
    for (int i = 0; i < steps; ++i) {
        double nx = cx + sx, ny = cy + sy;
        int ocx = static_cast<int>(cx), ocy = static_cast<int>(cy);
        int ncx = static_cast<int>(nx), ncy = static_cast<int>(ny);
        if (ncx < 0 || ncx >= grid_size || ncy < 0 || ncy >= grid_size) return false;
        if (g[ncy][ncx] != 0) return false;
        if (ncx != ocx || ncy != ocy) {
            if (std::abs(ncx - ocx) + std::abs(ncy - ocy) > 1) return false; // cut a corner
            if (conn[{ocx, ocy}].find({ncx, ncy}) == conn[{ocx, ocy}].end()) return false;
        }
        cx = nx; cy = ny;
    }
    return true;
}

Monster* World::find_monster(int id) {
    for (auto& m : monsters) if (m.id == id) return &m;
    return nullptr;
}

const Monster* World::find_monster(int id) const {
    for (const auto& m : monsters) if (m.id == id) return &m;
    return nullptr;
}

int World::boss_count() const {
    int n = 0;
    for (const auto& m : monsters) if (m.type == 2) n++;
    return n;
}

bool World::damage_monster(int id, int dmg) {
    for (auto it = monsters.begin(); it != monsters.end(); ++it) {
        if (it->id != id) continue;
        it->hp -= dmg;
        it->hit_flash = 8; // briefly highlight red on a successful hit
        if (it->hp > 0) return false;
        monsters.erase(it);
        return true;
    }
    return false;
}

std::optional<std::vector<World::Cell>> World::find_path() {
    std::set<Cell> visited;
    std::queue<Cell> queue;
    std::map<Cell, Cell> parent;
    queue.push(start);
    visited.insert(start);
    parent[start] = start;
    while (!queue.empty()) {
        auto pos = queue.front();
        queue.pop();
        if (pos == end) {
            std::vector<Cell> path;
            auto current = end;
            while (current != start) {
                path.push_back(current);
                current = parent[current];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }
        for (const auto& neigh : connections[pos]) {
            if (visited.find(neigh) == visited.end()) {
                visited.insert(neigh);
                parent[neigh] = pos;
                queue.push(neigh);
            }
        }
    }
    return std::nullopt;
}

void World::generate_maze() {
    int max_attempts = 10;
    int attempt = 0;
    bool solvable = false;
    while (attempt < max_attempts) {
        grid.assign(grid_size, std::vector<int>(grid_size, 1));
        connections.clear();
        start = { 0, 0 };
        int half_size = grid_size / 2;
        int end_x = rng.range(half_size, grid_size - 1);
        int end_y = rng.range(half_size, grid_size - 1);
        end = { end_x, end_y };
        std::set<Cell> visited;
        grid[0][0] = 0;
        visited.insert(start);
        std::stack<std::pair<Cell, Cell>> stack;
        stack.push({ start, {0, 0} });

        const std::vector<Cell> directions = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
        const double extra_branch_prob = 0.3;

        while (!stack.empty()) {
            auto [pos, in_dir] = stack.top();
            int x = pos.first, y = pos.second;
            std::vector<std::tuple<int, int, Cell>> unvisited_neighbors;
            for (const auto& d : directions) {
                int nx = x + d.first, ny = y + d.second;
                if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && visited.find({ nx, ny }) == visited.end()) {
                    unvisited_neighbors.emplace_back(nx, ny, d);
                }
            }
            if (!unvisited_neighbors.empty()) {
                // Lean toward the exit, and more so toward carrying straight on. Whole-number
                // weights, so every platform picks alike.
                std::vector<uint32_t> weights;
                const int max_dist = grid_size * 2;
                for (const auto& [nx, ny, d] : unvisited_neighbors) {
                    int dist = std::abs(nx - end.first) + std::abs(ny - end.second);
                    uint32_t w = static_cast<uint32_t>((max_dist - dist) * (max_dist - dist));
                    if (d == in_dir) w *= 5;
                    weights.push_back(std::max<uint32_t>(w, 1));
                }
                size_t chosen_idx = rng.weighted(weights);
                auto [nx, ny, d] = unvisited_neighbors[chosen_idx];
                connections[pos].insert({ nx, ny });
                connections[{nx, ny}].insert(pos);
                grid[ny][nx] = 0;
                visited.insert({ nx, ny });
                stack.push({ {nx, ny}, d });

                unvisited_neighbors.erase(unvisited_neighbors.begin() + chosen_idx);
                weights.erase(weights.begin() + chosen_idx);

                if (rng.chance(extra_branch_prob) && !unvisited_neighbors.empty()) {
                    size_t extra_idx = static_cast<size_t>(rng.range(0, static_cast<int>(unvisited_neighbors.size()) - 1));
                    auto [ex, ey, ed] = unvisited_neighbors[extra_idx];
                    if (visited.find({ ex, ey }) == visited.end()) {
                        connections[pos].insert({ ex, ey });
                        connections[{ex, ey}].insert(pos);
                        grid[ey][ex] = 0;
                        visited.insert({ ex, ey });
                        stack.push({ {ex, ey}, ed });
                    }
                }
            }
            else {
                stack.pop();
            }
        }

        auto path = find_path();
        if (!path) {
            int cx = start.first, cy = start.second;
            while (cx != end.first) {
                int step = (end.first > cx) ? 1 : -1;
                int nx = cx + step;
                if (nx >= 0 && nx < grid_size && grid[cy][nx] == 1) {
                    grid[cy][nx] = 0;
                    connections[{cx, cy}].insert({ nx, cy });
                    connections[{nx, cy}].insert({ cx, cy });
                }
                cx = nx;
            }
            while (cy != end.second) {
                int step = (end.second > cy) ? 1 : -1;
                int ny = cy + step;
                if (ny >= 0 && ny < grid_size && grid[ny][cx] == 1) {
                    grid[ny][cx] = 0;
                    connections[{cx, cy}].insert({ cx, ny });
                    connections[{cx, ny}].insert({ cx, cy });
                }
                cy = ny;
            }
        }

        rooms.clear();
        int num_rooms = rng.range(1, 3);
        for (int r = 0; r < num_rooms; r++) {
            int room_size = rng.range(3, 5);
            int max_start = grid_size - room_size;
            int rx = rng.range(0, max_start);
            int ry = rng.range(0, max_start);
            rooms.emplace_back(rx, ry, room_size);
            for (int dy = 0; dy < room_size; dy++) {
                for (int dx = 0; dx < room_size; dx++) {
                    grid[ry + dy][rx + dx] = 0;
                    if (dx < room_size - 1) {
                        connections[{rx + dx, ry + dy}].insert({ rx + dx + 1, ry + dy });
                        connections[{rx + dx + 1, ry + dy}].insert({ rx + dx, ry + dy });
                    }
                    if (dy < room_size - 1) {
                        connections[{rx + dx, ry + dy}].insert({ rx + dx, ry + dy + 1 });
                        connections[{rx + dx, ry + dy + 1}].insert({ rx + dx, ry + dy });
                    }
                }
            }
        }

        auto original_path = find_path();
        size_t original_len = original_path ? original_path->size() : 0;

        std::vector<std::pair<Cell, Cell>> possible_edges;
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                if (grid[y][x] == 0) {
                    for (const auto& d : directions) {
                        int nx = x + d.first, ny = y + d.second;
                        if (nx > x || (nx == x && ny > y)) {
                            if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && grid[ny][nx] == 0) {
                                if (connections[{x, y}].find({ nx, ny }) == connections[{x, y}].end()) {
                                    possible_edges.emplace_back(std::make_pair(x, y), std::make_pair(nx, ny));
                                }
                            }
                        }
                    }
                }
            }
        }

        rng.shuffle(possible_edges);
        int added_loops = 0;
        const int max_loops = 10;
        for (auto& edge : possible_edges) {
            auto u = edge.first, v = edge.second;
            connections[u].insert(v);
            connections[v].insert(u);
            auto new_path = find_path();
            size_t new_len = new_path ? new_path->size() : 0;
            if (new_path && new_len >= original_len) {
                added_loops++;
                if (added_loops >= max_loops) break;
            }
            else {
                connections[u].erase(v);
                connections[v].erase(u);
                if (connections[u].empty()) connections.erase(u);
                if (connections[v].empty()) connections.erase(v);
            }
        }

        if (find_path()) {
            solvable = true;
            break;
        }
        attempt++;
    }

    if (!solvable) {
        std::cerr << "Failed to generate solvable maze" << std::endl;
        std::exit(-1);
    }
}

// The lower level is the opposite of the maze: everything is open floor, and only a
// handful of long straight walls break it into a few huge rooms. Each wall gets a
// doorway so the whole floor stays walkable.
void World::generate_lower_level() {
    lower_grid.assign(grid_size, std::vector<int>(grid_size, 0));
    lower_connections.clear();
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            if (x + 1 < grid_size) link(lower_connections, { x, y }, { x + 1, y });
            if (y + 1 < grid_size) link(lower_connections, { x, y }, { x, y + 1 });
        }
    }

    int num_walls = rng.range(4, 6);
    for (int w = 0; w < num_walls; w++) {
        bool vertical = rng.range(0, 1) == 0;
        // The grid line the wall sits on, i.e. the seam between cells line-1 and line.
        int line = rng.range(5, grid_size - 5);
        int span = rng.range(12, 22);
        int span_start = rng.range(0, grid_size - span);
        int gap = span_start + span / 2; // a two-cell doorway through the middle
        for (int i = span_start; i < span_start + span; i++) {
            if (i == gap || i == gap + 1) continue;
            if (vertical) unlink(lower_connections, { line - 1, i }, { line, i });
            else          unlink(lower_connections, { i, line - 1 }, { i, line });
        }
    }
}

// Cut a straight staircase into the maze a couple of tiles in front of the spawn, running
// down to the lower level. The run belongs to both levels: it is chained together in each
// connection map, but only its top end opens onto the maze and only its bottom end opens
// onto the lower floor, so walking through it is the one way between the two.
void World::carve_staircase() {
    const int stair_len = 5;
    int sx = start.first + 2, sy = start.second;
    stair_cells.clear();
    for (int i = 0; i < stair_len; i++) stair_cells.emplace_back(sx + i, sy);

    // Approach corridor from the spawn to the head of the stairs. Carving only ever opens
    // cells, so it cannot make the maze unsolvable.
    for (int x = start.first; x < sx; x++) {
        grid[sy][x] = 0;
        grid[sy][x + 1] = 0;
        link(connections, { x, sy }, { x + 1, sy });
    }

    for (const auto& c : stair_cells) grid[c.second][c.first] = 0;
    for (size_t i = 0; i + 1 < stair_cells.size(); i++) {
        link(connections, stair_cells[i], stair_cells[i + 1]);
        link(lower_connections, stair_cells[i], stair_cells[i + 1]);
    }

    // Seal every other edge of the shaft on both levels so you cannot step off mid-descent.
    const int dxs[4] = { 1, -1, 0, 0 };
    const int dys[4] = { 0, 0, 1, -1 };
    for (const auto& c : stair_cells) {
        for (int d = 0; d < 4; d++) {
            Cell n{ c.first + dxs[d], c.second + dys[d] };
            if (n.first < 0 || n.first >= grid_size || n.second < 0 || n.second >= grid_size) continue;
            if (is_stair_cell(n.first, n.second)) continue;
            unlink(connections, c, n);
            unlink(lower_connections, c, n);
        }
    }

    // Top of the stairs opens onto the maze; the bottom opens onto the lower floor.
    link(connections, stair_cells.front(), { sx - 1, sy });
    if (sx + stair_len < grid_size)
        link(lower_connections, stair_cells.back(), { sx + stair_len, sy });

    // Sealing the shaft cut every maze edge that ran into it, which could otherwise strand
    // whatever hung off those cells - possibly the route to the exit. A corridor alongside
    // the stairwell reconnects all of them: every neighbour the shaft lost now sits on it.
    int bypass = sy + 1;
    int bypass_end = std::min(sx + stair_len, grid_size - 1);
    for (int x = start.first + 1; x <= bypass_end; x++) {
        grid[bypass][x] = 0;
        if (x > start.first + 1) link(connections, { x - 1, bypass }, { x, bypass });
    }
    link(connections, { start.first + 1, sy }, { start.first + 1, bypass });
    grid[sy][bypass_end] = 0;
    link(connections, { bypass_end, bypass }, { bypass_end, sy });
}

void World::spawn_monsters() {
    monsters.clear();
    std::vector<Cell> path_cells;
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            if (grid[y][x] == 0 && Cell{x, y} != start && Cell{x, y} != end && !is_stair_cell(x, y)) {
                bool in_room = false;
                for (const auto& room : rooms) {
                    int rx, ry, rs;
                    std::tie(rx, ry, rs) = room;
                    if (rx <= x && x < rx + rs && ry <= y && y < ry + rs) {
                        in_room = true;
                        break;
                    }
                }
                if (!in_room) path_cells.emplace_back(x, y);
            }
        }
    }
    rng.shuffle(path_cells);
    int num_monsters = rng.range(5, 12);
    for (int i = 0; i < std::min(num_monsters, static_cast<int>(path_cells.size())); i++) {
        int mx = path_cells[i].first, my = path_cells[i].second;
        int hp = 300; // 3 body shots (100 each) or 1 headshot (300) to kill
        int cooldown = rng.range(0, 180);
        monsters.push_back({ mx + 0.5, my + 0.5, hp, 1, mx + 0.5, my + 0.5, cooldown });
    }
    for (const auto& room : rooms) {
        int rx, ry, rs;
        std::tie(rx, ry, rs) = room;
        int mx = rx + rs / 2;
        int my = ry + rs / 2;
        int boss_hp = rng.range(310, 440); // 7-9 body shots (50 each) or 3 headshots (150) to kill
        int cooldown = rng.range(0, 60);
        monsters.push_back({ mx + 0.5, my + 0.5, boss_hp, 2, mx + 0.5, my + 0.5, cooldown });
    }
    // Ids follow spawn order, so every machine that built this maze agrees on them.
    for (size_t i = 0; i < monsters.size(); i++) {
        monsters[i].id = static_cast<int>(i);
        monsters[i].net_x = monsters[i].x;
        monsters[i].net_y = monsters[i].y;
    }
}

void World::spawn_health_packs() {
    health_packs.clear();
    std::vector<Cell> path_cells;
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            if (grid[y][x] == 0 && Cell{x, y} != start && Cell{x, y} != end && !is_stair_cell(x, y))
                path_cells.emplace_back(x, y);
        }
    }
    rng.shuffle(path_cells);
    for (int i = 0; i < 2 && i < static_cast<int>(path_cells.size()); i++) {
        int hx = path_cells[i].first, hy = path_cells[i].second;
        double px = hx + rng.real(0.2, 0.8);
        double py = hy + rng.real(0.2, 0.8);
        health_packs.push_back({ px, py, 0 });
    }

    // Two more on the lower level, anywhere on its open floor except the staircase itself.
    std::vector<Cell> lower_cells;
    for (int y = 0; y < grid_size; y++)
        for (int x = 0; x < grid_size; x++)
            if (lower_grid[y][x] == 0 && !is_stair_cell(x, y)) lower_cells.emplace_back(x, y);
    rng.shuffle(lower_cells);
    for (int i = 0; i < 2 && i < static_cast<int>(lower_cells.size()); i++) {
        int hx = lower_cells[i].first, hy = lower_cells[i].second;
        double px = hx + rng.real(0.2, 0.8);
        double py = hy + rng.real(0.2, 0.8);
        health_packs.push_back({ px, py, 1 });
    }
    for (size_t i = 0; i < health_packs.size(); i++) health_packs[i].id = static_cast<int>(i);
}

void World::step_monsters(double delta, int driven_boss, const std::vector<Target>& targets,
                          std::vector<Projectile>& shots) {
    const int dxs[4] = { -1, 1, 0, 0 };
    const int dys[4] = { 0, 0, -1, 1 };
    for (auto& m : monsters) {
        if (m.hit_flash > 0) m.hit_flash--; // fade out the red hit highlight
        if (m.id == driven_boss) {
            double k = std::min(1.0, delta * 15.0);
            m.x += (m.net_x - m.x) * k;
            m.y += (m.net_y - m.y) * k;
            m.target_x = m.x;
            m.target_y = m.y;
            continue;
        }
        double dx = m.target_x - m.x;
        double dy = m.target_y - m.y;
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 0.01) {
            int cx = static_cast<int>(m.x);
            int cy = static_cast<int>(m.y);
            std::vector<std::pair<double, double>> possible_targets;
            for (int i = 0; i < 4; i++) {
                int nx = cx + dxs[i], ny = cy + dys[i];
                // Never wander onto the staircase: monsters have no notion of height and
                // would float above the steps.
                if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && grid[ny][nx] == 0
                    && connections[{cx, cy}].count({ nx, ny }) && !is_stair_cell(nx, ny)) {
                    possible_targets.emplace_back(nx + 0.5, ny + 0.5);
                }
            }
            if (!possible_targets.empty()) {
                auto target = possible_targets[static_cast<size_t>(rng.range(0, static_cast<int>(possible_targets.size()) - 1))];
                m.target_x = target.first;
                m.target_y = target.second;
            }
        }
        else {
            double speed = (m.type == 1 ? 0.0072 : 0.01275) * delta * 60.0; // monster -28%, boss -15%
            m.x += (dx / dist) * speed;
            m.y += (dy / dist) * speed;
        }

        m.cooldown -= 1;
        if (m.cooldown <= 0) {
            // Fire at the nearest explorer on the maze level. With nobody there the shot
            // could not reach, so hold fire - but still reset the cooldown, so they don't all
            // volley the moment someone comes back up.
            const Target* best = nullptr;
            double best_dist = 1e18;
            for (const auto& t : targets) {
                double d = std::hypot(t.x - m.x, t.y - m.y);
                if (d > 0.0 && d < best_dist) { best_dist = d; best = &t; }
            }
            if (best) {
                bool is_boss = (m.type == 2);
                Projectile mp;
                mp.x = m.x; mp.y = m.y;
                mp.dir_x = (best->x - m.x) / best_dist;
                mp.dir_y = (best->y - m.y) / best_dist;
                mp.speed = is_boss ? 0.0714 : 0.075; // boss projectiles slower than regular ones
                mp.from_boss = is_boss;
                mp.owner = 0;
                shots.push_back(mp);
            }
            m.cooldown = (m.type == 2 ? 86 : 300); // boss fires 30% less often, monsters 40% less
        }
    }
}

uint64_t World::world_checksum() const {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](int64_t v) { h ^= static_cast<uint64_t>(v); h *= 1099511628211ull; };
    auto mix_conn = [&](const Connections& conn) {
        for (const auto& [cell, links] : conn) {
            if (links.empty()) continue; // empty entries only exist from lookups, not layout
            mix(cell.first * 64 + cell.second);
            for (const auto& n : links) mix(10000 + n.first * 64 + n.second);
        }
    };
    for (const auto& row : grid) for (int c : row) mix(c);
    mix_conn(connections);
    mix_conn(lower_connections);
    mix(start.first); mix(start.second); mix(end.first); mix(end.second);
    for (const auto& m : monsters) {
        mix(m.id); mix(m.type); mix(m.hp);
        mix(std::llround(m.x * 1000.0)); mix(std::llround(m.y * 1000.0));
    }
    for (const auto& p : health_packs) {
        mix(p.id); mix(p.level);
        mix(std::llround(p.x * 1000.0)); mix(std::llround(p.y * 1000.0));
    }
    return h;
}
