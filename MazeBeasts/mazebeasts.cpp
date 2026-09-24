#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <stack>
#include <queue>
#include <random>
#include <chrono>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <utility>
#include <fstream>
#include <string>
#include <optional>
#include <cstring>
#include <cstdint>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

// Single-header audio playback (decodes FLAC/WAV/MP3). Place miniaudio.h beside the stb_ headers.
// The implementation section includes <windows.h>. Guard against its min/max macros (which would
// otherwise turn std::min/std::max into "std::(" errors) and silence miniaudio's internal warnings.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifdef APIENTRY
#undef APIENTRY   // avoid redefinition warning vs. glad/GLFW
#endif
#define MINIAUDIO_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable: 4244 4267 4005 4996) // conversion / macro-redef / deprecated inside miniaudio
#include "miniaudio.h"
#pragma warning(pop)

// Every texture, sound and the font live in one assets.dat. Entries keep their original
// filenames, so call sites still ask for "wall_texture.png" and don't care where it came from.
//
//   "MBPACK01"  magic (8 bytes)
//   uint32      entry count
//   per entry:  uint16 nameLen, name bytes, uint64 offset, uint64 size
//   then the raw blobs.
class AssetPack {
public:
    bool load(const char* path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        blob.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (blob.size() < 12 || std::memcmp(blob.data(), "MBPACK01", 8) != 0) {
            blob.clear();
            return false;
        }
        const unsigned char* p = blob.data() + 8;
        uint32_t count;
        std::memcpy(&count, p, 4); p += 4;
        for (uint32_t i = 0; i < count; i++) {
            uint16_t name_len;
            std::memcpy(&name_len, p, 2); p += 2;
            std::string name(reinterpret_cast<const char*>(p), name_len); p += name_len;
            uint64_t off, size;
            std::memcpy(&off, p, 8); p += 8;
            std::memcpy(&size, p, 8); p += 8;
            if (off + size > blob.size()) { blob.clear(); entries.clear(); return false; }
            entries[name] = { static_cast<size_t>(off), static_cast<size_t>(size) };
        }
        return true;
    }

    // Returns nullptr when the name isn't packed, so callers can fall back to a loose file.
    const unsigned char* find(const std::string& name, size_t* size) const {
        auto it = entries.find(name);
        if (it == entries.end()) return nullptr;
        *size = it->second.second;
        return blob.data() + it->second.first;
    }

    size_t count() const { return entries.size(); }

private:
    std::vector<unsigned char> blob;
    std::map<std::string, std::pair<size_t, size_t>> entries;
};

// miniaudio reaches for assets through a VFS, so serving the pack through one means
// ma_engine_play_sound() and ma_sound_init_from_file() keep taking plain filenames.
// Anything not in the pack is handed to miniaudio's own default (real file) VFS.
struct PackVFS {
    ma_vfs_callbacks cb;   // must stay first: miniaudio casts the struct to this
    ma_default_vfs fallback;
    AssetPack* pack;
};

struct PackFile {
    const unsigned char* data = nullptr; // set when served from the pack
    size_t size = 0;
    size_t cursor = 0;
    ma_vfs_file real = nullptr;          // set when delegating to the default VFS
};

static ma_result pack_vfs_open(ma_vfs* vfs, const char* path, ma_uint32 mode, ma_vfs_file* out) {
    PackVFS* self = reinterpret_cast<PackVFS*>(vfs);
    size_t size = 0;
    const unsigned char* data = (self->pack && (mode & MA_OPEN_MODE_READ))
        ? self->pack->find(path, &size) : nullptr;
    PackFile* f = new PackFile();
    if (data) {
        f->data = data;
        f->size = size;
    } else {
        if (ma_vfs_open(&self->fallback, path, mode, &f->real) != MA_SUCCESS) {
            delete f;
            return MA_DOES_NOT_EXIST;
        }
    }
    *out = f;
    return MA_SUCCESS;
}

static ma_result pack_vfs_open_w(ma_vfs* vfs, const wchar_t* path, ma_uint32 mode, ma_vfs_file* out) {
    std::string narrow;
    for (const wchar_t* p = path; *p; ++p) narrow.push_back(static_cast<char>(*p));
    return pack_vfs_open(vfs, narrow.c_str(), mode, out);
}

static ma_result pack_vfs_close(ma_vfs* vfs, ma_vfs_file file) {
    PackVFS* self = reinterpret_cast<PackVFS*>(vfs);
    PackFile* f = static_cast<PackFile*>(file);
    if (f->real) ma_vfs_close(&self->fallback, f->real);
    delete f;
    return MA_SUCCESS;
}

static ma_result pack_vfs_read(ma_vfs* vfs, ma_vfs_file file, void* dst, size_t bytes, size_t* read) {
    PackVFS* self = reinterpret_cast<PackVFS*>(vfs);
    PackFile* f = static_cast<PackFile*>(file);
    if (f->real) return ma_vfs_read(&self->fallback, f->real, dst, bytes, read);
    size_t avail = f->size - f->cursor;
    size_t n = bytes < avail ? bytes : avail;
    std::memcpy(dst, f->data + f->cursor, n);
    f->cursor += n;
    if (read) *read = n;
    return n == 0 && bytes > 0 ? MA_AT_END : MA_SUCCESS;
}

static ma_result pack_vfs_write(ma_vfs*, ma_vfs_file, const void*, size_t, size_t*) {
    return MA_NOT_IMPLEMENTED; // the pack is read-only
}

static ma_result pack_vfs_seek(ma_vfs* vfs, ma_vfs_file file, ma_int64 offset, ma_seek_origin origin) {
    PackVFS* self = reinterpret_cast<PackVFS*>(vfs);
    PackFile* f = static_cast<PackFile*>(file);
    if (f->real) return ma_vfs_seek(&self->fallback, f->real, offset, origin);
    ma_int64 base = origin == ma_seek_origin_current ? static_cast<ma_int64>(f->cursor)
                  : origin == ma_seek_origin_end     ? static_cast<ma_int64>(f->size)
                                                     : 0;
    ma_int64 target = base + offset;
    if (target < 0 || target > static_cast<ma_int64>(f->size)) return MA_INVALID_ARGS;
    f->cursor = static_cast<size_t>(target);
    return MA_SUCCESS;
}

static ma_result pack_vfs_tell(ma_vfs* vfs, ma_vfs_file file, ma_int64* cursor) {
    PackVFS* self = reinterpret_cast<PackVFS*>(vfs);
    PackFile* f = static_cast<PackFile*>(file);
    if (f->real) return ma_vfs_tell(&self->fallback, f->real, cursor);
    *cursor = static_cast<ma_int64>(f->cursor);
    return MA_SUCCESS;
}

static ma_result pack_vfs_info(ma_vfs* vfs, ma_vfs_file file, ma_file_info* info) {
    PackVFS* self = reinterpret_cast<PackVFS*>(vfs);
    PackFile* f = static_cast<PackFile*>(file);
    if (f->real) return ma_vfs_info(&self->fallback, f->real, info);
    info->sizeInBytes = f->size;
    return MA_SUCCESS;
}

struct Monster {
    double x, y;
    int hp;
    int type;
    double target_x, target_y;
    int cooldown;
    int hit_flash = 0; // frames remaining to render this monster tinted red after being hit
};

struct HealthPack {
    double x, y;
    int level = 0; // 0 = maze, 1 = lower level; both share the same x/y footprint
};

// Medpacks sit on the floor and are small enough to step around. The pickup radius is what
// makes that possible: packs spawn 0.2-0.8 across a cell and the player can get to within
// 0.06 of a wall, so even a pack dead-centre in a one-tile corridor leaves 0.22 of clearance.
constexpr float  MEDPACK_HALF_SIZE     = 0.14f;  // billboard half-extent (was 0.5, a full wall high)
constexpr double MEDPACK_PICKUP_RADIUS = 0.22;   // was 0.5, which filled the whole corridor

// Fraction of an image's height that is fully transparent below the figure. Sprite billboards
// are lowered by this much so figures stand on the floor rather than hovering over it.
// Rows are scanned from the bottom; stb_image is set to flip on load, so row 0 IS the bottom.
static float transparent_bottom_margin(const unsigned char* data, int w, int h, int channels) {
    if (channels != 4 || h <= 0) return 0.0f;
    for (int y = 0; y < h; y++) {
        const unsigned char* row = data + static_cast<size_t>(y) * w * 4;
        for (int x = 0; x < w; x++)
            if (row[x * 4 + 3] > 24) return static_cast<float>(y) / h;
    }
    return 0.0f;
}

struct Projectile {
    double x, y;
    double dir_x, dir_y;
    double speed;
    double z = 0.5;      // world height (upper level spans 0..1, lower level spans -2..0)
    double dir_z = 0.0;  // vertical component of travel direction
    bool from_boss = false; // boss shots hit harder and render smaller
    int level = 0;       // which level's walls this shot collides with (0 = maze, 1 = lower)
};

struct Sprite {
    glm::vec3 pos;
    float size;
    GLuint tex;
    glm::vec4 color;
    bool use_tex;
    double dist;
};

class MazeGame {
private:
    int grid_size = 33;
    std::vector<std::vector<int>> grid;
    std::map<std::pair<int, int>, std::set<std::pair<int, int>>> connections;
    std::pair<int, int> start, end;
    std::vector<std::tuple<int, int, int>> rooms;
    double player_pos_x, player_pos_y;
    double dir_x, dir_y;
    std::vector<Monster> monsters;
    std::vector<HealthPack> health_packs;
    std::vector<Projectile> projectiles;
    std::vector<Projectile> monster_projectiles;
    int max_hp = 20;            // boss shots deal 4-5, so it takes 4-5 of them to kill
    int player_hp = max_hp;
    int damage_cooldown = 0;
    double min_dist = 0.06; // get right up to a wall, but a hair in front so you can't see past it

    // Lower level: a wide-open floor beneath the maze, reached by a staircase near the start.
    // It shares the maze's grid footprint, so (x, y) coordinates carry across both levels.
    static constexpr double lower_floor_y = -2.0; // world height of the lower floor
    std::vector<std::vector<int>> lower_grid;
    std::map<std::pair<int, int>, std::set<std::pair<int, int>>> lower_connections;
    std::vector<std::pair<int, int>> stair_cells; // the staircase run, ordered top to bottom
    int player_level = 0; // 0 = maze, 1 = lower level

    // Jumping (spacebar): height above whatever floor is under the player, and vertical speed.
    // Peak is v^2 / 2g = 0.3 units, which keeps your eye below the maze's 1-unit ceiling.
    static constexpr double JUMP_SPEED = 2.4;  // units/s at take-off
    static constexpr double GRAVITY    = 9.6;  // units/s^2 -> about half a second in the air
    double jump_height = 0.0;
    double jump_velocity = 0.0;

    GLFWwindow* window;
    int screen_width = 1280;
    int screen_height = 800;
    GLuint wall_tex, boundary_tex, monster_tex, monster2_tex, medpack_tex;
    // Empty strip under each sprite figure, as a fraction of image height (see load_textures).
    float monster_bottom_margin = 0.0f, boss_bottom_margin = 0.0f, medpack_bottom_margin = 0.0f;
    GLuint lower_wall_tex;  // rock face for the lower level's walls
    GLuint lower_floor_tex; // rock floor for the lower level (0 = fall back to a flat colour)
    float lower_floor_aspect = 1.0f; // width/height of that image, so its tiling stays square
    GLuint shader_program;

    // For walls
    GLuint wall_vao, wall_vbo, wall_ebo;
    GLuint boundary_vao, boundary_vbo, boundary_ebo;
    size_t wall_index_count, boundary_index_count;

    // For floor and ceiling. The upper floor is one quad per cell so the staircase can
    // leave a shaft open through it; the lower floor is a single slab.
    GLuint floor_vao, floor_vbo, floor_ebo;
    GLuint ceiling_vao, ceiling_vbo, ceiling_ebo;
    size_t floor_index_count;
    GLuint lower_floor_vao, lower_floor_vbo, lower_floor_ebo;

    // Lower-level walls and the staircase itself
    GLuint lower_wall_vao, lower_wall_vbo, lower_wall_ebo;
    GLuint lower_boundary_vao, lower_boundary_vbo, lower_boundary_ebo;
    size_t lower_wall_index_count, lower_boundary_index_count;
    GLuint stair_vao, stair_vbo, stair_ebo;
    size_t stair_index_count;

    // For minimap lines
    GLuint mini_wall_vao, mini_wall_vbo;
    size_t mini_wall_vertex_count;
    GLuint mini_lower_vao, mini_lower_vbo;
    size_t mini_lower_vertex_count;

    // First-person viewmodel. The gun is built from boxes in code and drawn unlit, so each
    // face's brightness is baked in at build time and faces are grouped by resulting colour.
    struct GunSection { size_t offset; size_t count; glm::vec4 color; };
    GLuint gun_vao, gun_vbo, gun_ebo;
    std::vector<GunSection> gun_sections;

    // For drawing
    GLuint quad_vao, quad_vbo;
    GLuint line_vao, line_vbo;
    GLuint circle_vao, circle_vbo;

    // Mesh data
    std::vector<float> wall_vertices;
    std::vector<float> boundary_vertices;
    std::vector<float> mini_wall_verts;
    std::vector<float> mini_lower_verts;

    // For text
    stbtt_fontinfo font_info;
    GLuint font_tex;
    unsigned char* font_bitmap;
    int font_bitmap_w = 512, font_bitmap_h = 512;
    stbtt_bakedchar cdata[96]; // ASCII 32..126

    double recoil = 0.0;        // counts down after each shot; drives the viewmodel kick only
    double recoil_time = 0.09;  // how long that kick takes to settle
    bool fire_pressed = false;  // previous mouse state, for edge-detected firing
    double last_time = 0.0;
    double yaw = 0.0; // for mouse look (horizontal, degrees)
    double pitch = 0.0; // vertical look/aim (degrees, + = up)
    bool first_mouse = true; // skip the first mouse delta so the view starts straight forward
    bool tab_pressed = false; // previous Tab key state, for edge-detected toggling
    bool tab_view = false;    // toggled map view (Tab), no longer hold-to-view
    bool dev_mode = false;    // developer mode (F5), enables F6 boss-room teleport
    bool f5_pressed = false;  // previous F5 key state
    bool f6_pressed = false;  // previous F6 key state
    bool f7_pressed = false;  // previous F7 key state
    size_t dev_tp_index = 0;  // which boss room the next F6 teleport targets
    std::mt19937 rng;
    double win_timer = 0.0;
    bool showing_win = false;
    double die_timer = 0.0;
    bool showing_die = false;

    // Assets
    AssetPack assets;
    PackVFS pack_vfs{};

    // Audio
    ma_engine sound_engine;
    ma_resource_manager resource_manager;
    bool resource_manager_ready = false;
    bool sound_ready = false;
    bool monster_in_view = false;   // true while a regular monster (not a boss) is visible on screen
    int inside_boss_room = -1;   // index of the boss room the player is currently in (-1 = none)
    ma_sound boss_sound;         // looping cue kept as a handle so we can stop it
    bool boss_sound_active = false;

public:
    MazeGame() : rng(static_cast<std::mt19937::result_type>(
        std::chrono::steady_clock::now().time_since_epoch().count())) {
        // Load the bundle first: textures, the font and audio all read through it.
        // Without it every lookup falls back to loose files, so a dev checkout still runs.
        if (assets.load("assets.dat"))
            std::cout << "assets.dat: " << assets.count() << " entries" << std::endl;
        else
            std::cerr << "assets.dat not found; falling back to loose files." << std::endl;

        init_glfw();
        init_glad();
        init_opengl();
        init_audio();
        init_font();
        init_draw_buffers();
        build_gun_mesh();
        load_textures();
        generate_maze();
        generate_lower_level();
        carve_staircase();
        player_pos_x = start.first + 0.5;
        player_pos_y = start.second + 0.5;
        dir_x = 1.0;
        dir_y = 0.0;
        yaw = 0.0;
        pitch = 0.0;
        spawn_monsters();
        spawn_health_packs();
        build_meshes();
        run();
    }

    ~MazeGame() {
        glDeleteVertexArrays(1, &wall_vao);
        glDeleteBuffers(1, &wall_vbo);
        glDeleteBuffers(1, &wall_ebo);
        glDeleteVertexArrays(1, &boundary_vao);
        glDeleteBuffers(1, &boundary_vbo);
        glDeleteBuffers(1, &boundary_ebo);
        glDeleteVertexArrays(1, &floor_vao);
        glDeleteBuffers(1, &floor_vbo);
        glDeleteBuffers(1, &floor_ebo);
        glDeleteVertexArrays(1, &ceiling_vao);
        glDeleteBuffers(1, &ceiling_vbo);
        glDeleteBuffers(1, &ceiling_ebo);
        glDeleteVertexArrays(1, &lower_wall_vao);
        glDeleteBuffers(1, &lower_wall_vbo);
        glDeleteBuffers(1, &lower_wall_ebo);
        glDeleteVertexArrays(1, &lower_boundary_vao);
        glDeleteBuffers(1, &lower_boundary_vbo);
        glDeleteBuffers(1, &lower_boundary_ebo);
        glDeleteVertexArrays(1, &lower_floor_vao);
        glDeleteBuffers(1, &lower_floor_vbo);
        glDeleteBuffers(1, &lower_floor_ebo);
        glDeleteVertexArrays(1, &stair_vao);
        glDeleteBuffers(1, &stair_vbo);
        glDeleteBuffers(1, &stair_ebo);
        glDeleteVertexArrays(1, &mini_wall_vao);
        glDeleteBuffers(1, &mini_wall_vbo);
        glDeleteVertexArrays(1, &mini_lower_vao);
        glDeleteBuffers(1, &mini_lower_vbo);
        glDeleteVertexArrays(1, &gun_vao);
        glDeleteBuffers(1, &gun_vbo);
        glDeleteBuffers(1, &gun_ebo);
        glDeleteVertexArrays(1, &quad_vao);
        glDeleteBuffers(1, &quad_vbo);
        glDeleteVertexArrays(1, &line_vao);
        glDeleteBuffers(1, &line_vbo);
        glDeleteVertexArrays(1, &circle_vao);
        glDeleteBuffers(1, &circle_vbo);
        glDeleteProgram(shader_program);
        glDeleteTextures(1, &font_tex);
        delete[] font_bitmap;
        stop_boss_sound();
        if (sound_ready) ma_engine_uninit(&sound_engine);
        if (resource_manager_ready) ma_resource_manager_uninit(&resource_manager);
        glfwTerminate();
    }

    void init_audio() {
        // Route miniaudio's file access through the pack so the play calls below can keep
        // using plain filenames.
        pack_vfs.cb.onOpen  = pack_vfs_open;
        pack_vfs.cb.onOpenW = pack_vfs_open_w;
        pack_vfs.cb.onClose = pack_vfs_close;
        pack_vfs.cb.onRead  = pack_vfs_read;
        pack_vfs.cb.onWrite = pack_vfs_write;
        pack_vfs.cb.onSeek  = pack_vfs_seek;
        pack_vfs.cb.onTell  = pack_vfs_tell;
        pack_vfs.cb.onInfo  = pack_vfs_info;
        ma_default_vfs_init(&pack_vfs.fallback, nullptr);
        pack_vfs.pack = &assets;

        ma_resource_manager_config rmc = ma_resource_manager_config_init();
        rmc.pVFS = &pack_vfs;
        if (ma_resource_manager_init(&rmc, &resource_manager) != MA_SUCCESS) {
            std::cerr << "Failed to initialize audio resource manager; sound disabled." << std::endl;
            sound_ready = false;
            return;
        }
        resource_manager_ready = true;

        ma_engine_config ec = ma_engine_config_init();
        ec.pResourceManager = &resource_manager;
        if (ma_engine_init(&ec, &sound_engine) != MA_SUCCESS) {
            std::cerr << "Failed to initialize audio engine; sound disabled." << std::endl;
            sound_ready = false;
        } else {
            sound_ready = true;
        }
    }

    void play_sound(const char* file) {
        if (sound_ready) ma_engine_play_sound(&sound_engine, file, nullptr);
    }

    // Boss cue loops while the player is in a boss room; kept as a handle so it can be stopped.
    void start_boss_sound() {
        if (!sound_ready || boss_sound_active) return;
        if (ma_sound_init_from_file(&sound_engine, "boss_sound.flac", 0, nullptr, nullptr, &boss_sound) != MA_SUCCESS) return;
        ma_sound_set_looping(&boss_sound, MA_TRUE);
        ma_sound_start(&boss_sound);
        boss_sound_active = true;
    }

    void stop_boss_sound() {
        if (!boss_sound_active) return;
        ma_sound_stop(&boss_sound);
        ma_sound_uninit(&boss_sound);
        boss_sound_active = false;
    }

    // How far a monster's 1-unit-tall billboard is dropped so the figure's feet touch the floor.
    float monster_sink(int type) const {
        return type == 2 ? boss_bottom_margin : monster_bottom_margin;
    }

    bool room_has_boss(int room_idx) {
        int rx, ry, rs;
        std::tie(rx, ry, rs) = rooms[room_idx];
        for (const auto& m : monsters) {
            if (m.type == 2 && m.x >= rx && m.x < rx + rs && m.y >= ry && m.y < ry + rs) return true;
        }
        return false;
    }

    // --- Level plumbing -------------------------------------------------------------------
    // Both levels share the same coordinate space, so collision, line of sight and the map
    // all just need to be told which level's grid to consult.

    const std::vector<std::vector<int>>& grid_for(int level) const {
        return level == 0 ? grid : lower_grid;
    }

    std::map<std::pair<int, int>, std::set<std::pair<int, int>>>& conn_for(int level) {
        return level == 0 ? connections : lower_connections;
    }

    static void link(std::map<std::pair<int, int>, std::set<std::pair<int, int>>>& conn,
                     std::pair<int, int> a, std::pair<int, int> b) {
        conn[a].insert(b);
        conn[b].insert(a);
    }

    static void unlink(std::map<std::pair<int, int>, std::set<std::pair<int, int>>>& conn,
                       std::pair<int, int> a, std::pair<int, int> b) {
        conn[a].erase(b);
        conn[b].erase(a);
    }

    bool is_stair_cell(int x, int y) const {
        for (const auto& c : stair_cells) if (c.first == x && c.second == y) return true;
        return false;
    }

    // How far along the staircase a position is: 0 at the top step, 1 at the bottom.
    // The run is laid out along +x, so only the x coordinate matters.
    double stair_progress(double x) const {
        if (stair_cells.empty()) return 0.0;
        double t = (x - stair_cells.front().first) / static_cast<double>(stair_cells.size());
        return std::clamp(t, 0.0, 1.0);
    }

    // World height of the surface under a position. On the staircase this ramps smoothly
    // between the two floors; everywhere else it is whichever floor the player is on.
    double floor_y_at(double x, double y) const {
        if (is_stair_cell(static_cast<int>(x), static_cast<int>(y)))
            return stair_progress(x) * lower_floor_y;
        return player_level == 0 ? 0.0 : lower_floor_y;
    }

    double player_eye_y() const { return floor_y_at(player_pos_x, player_pos_y) + 0.5 + jump_height; }

    // The staircase cells belong to both levels, so which level the player counts as being on
    // is decided by how far down the stairs they are. By the time they can step off either end
    // the answer has already settled, which is what makes the hand-off seamless.
    void update_player_level() {
        if (is_stair_cell(static_cast<int>(player_pos_x), static_cast<int>(player_pos_y)))
            player_level = stair_progress(player_pos_x) > 0.5 ? 1 : 0;
    }

    // True if nothing blocks a straight line between two points in the maze (grid + connections).
    bool has_line_of_sight(double x0, double y0, double x1, double y1) {
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
            if (grid[ncy][ncx] != 0) return false;
            if (ncx != ocx || ncy != ocy) {
                if (std::abs(ncx - ocx) + std::abs(ncy - ocy) > 1) return false; // cut a corner
                if (connections[{ocx, ocy}].find({ncx, ncy}) == connections[{ocx, ocy}].end()) return false;
            }
            cx = nx; cy = ny;
        }
        return true;
    }

    // A monster is "seen" when it is within the view cone and not hidden behind a wall.
    bool monster_visible(const Monster& m) {
        double dx = m.x - player_pos_x;
        double dy = m.y - player_pos_y;
        double dist = std::hypot(dx, dy);
        if (dist < 1e-6) return true;
        double nx = dx / dist, ny = dy / dist;
        // Roughly the on-screen horizontal field of view (~40 deg half-angle).
        if (nx * dir_x + ny * dir_y < std::cos(glm::radians(40.0))) return false;
        return has_line_of_sight(player_pos_x, player_pos_y, m.x, m.y);
    }

    void update_audio_cues() {
        // Nothing to hear on the lower level: it has no monsters and no boss rooms.
        if (player_level != 0) {
            monster_in_view = false;
            inside_boss_room = -1;
            stop_boss_sound();
            return;
        }

        // Monster cue: play once when a regular monster (type 1, not a boss) comes into view.
        bool any_visible = false;
        for (const auto& m : monsters) {
            if (m.type == 1 && monster_visible(m)) { any_visible = true; break; }
        }
        if (any_visible && !monster_in_view) {
            monster_in_view = true;
            play_sound("monster_sound.flac");
        } else if (!any_visible) {
            monster_in_view = false;
        }

        // Boss cue: loop boss_sound only when the player is actually near a living boss. It used
        // to start anywhere within 6 tiles of a boss room's edge, measured in a straight line
        // through walls - up to ~8.5 tiles away. It now keys off the distance to the boss itself.
        // Starting and stopping at different distances stops it stuttering on the boundary.
        const double boss_cue_start = 3.0;
        const double boss_cue_stop  = 4.0;
        int cur = -1;
        int px = static_cast<int>(player_pos_x);
        int py = static_cast<int>(player_pos_y);
        for (size_t i = 0; i < rooms.size(); ++i) {
            int rx, ry, rs;
            std::tie(rx, ry, rs) = rooms[i];
            if (px >= rx && px < rx + rs && py >= ry && py < ry + rs) cur = static_cast<int>(i);
        }
        inside_boss_room = cur;

        double nearest_boss = 1e9;
        for (const auto& m : monsters)
            if (m.type == 2) nearest_boss = std::min(nearest_boss, std::hypot(m.x - player_pos_x, m.y - player_pos_y));
        bool want_boss_sound = nearest_boss < (boss_sound_active ? boss_cue_stop : boss_cue_start);
        if (want_boss_sound) start_boss_sound();
        else stop_boss_sound();
    }

    void init_glfw() {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
        // Fullscreen windowed (borderless)
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        screen_width = mode->width;
        screen_height = mode->height;
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(screen_width, screen_height, "MazeBeasts - 3D", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(-1);
        }
        glfwSetWindowPos(window, 0, 0);
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetWindowUserPointer(window, this);
    }

    static void framebuffer_size_callback(GLFWwindow* w, int width, int height) {
        glViewport(0, 0, width, height);
        MazeGame* game = static_cast<MazeGame*>(glfwGetWindowUserPointer(w));
        game->screen_width = width;
        game->screen_height = height;
    }

    void init_glad() {
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            exit(-1);
        }
    }

    void init_opengl() {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        shader_program = create_shader();
    }

    GLuint create_shader() {
        const char* vert_src = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            TexCoord = aTexCoord;
        }
        )";

        const char* frag_src = R"(
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;
        uniform sampler2D texture1;
        uniform int use_texture;
        uniform vec4 color;
        void main() {
            if (use_texture == 1) {
                FragColor = texture(texture1, TexCoord) * color;
            } else {
                FragColor = color;
            }
        }
        )";

        GLuint vert = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert, 1, &vert_src, nullptr);
        glCompileShader(vert);
        check_compile(vert, "VERTEX");

        GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag, 1, &frag_src, nullptr);
        glCompileShader(frag);
        check_compile(frag, "FRAGMENT");

        GLuint program = glCreateProgram();
        glAttachShader(program, vert);
        glAttachShader(program, frag);
        glLinkProgram(program);
        check_link(program);

        glDeleteShader(vert);
        glDeleteShader(frag);
        return program;
    }

    void check_compile(GLuint shader, const char* type) {
        int success;
        char info[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, nullptr, info);
            std::cerr << "SHADER " << type << " COMPILATION FAILED\n" << info << std::endl;
        }
    }

    void check_link(GLuint program) {
        int success;
        char info[512];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, nullptr, info);
            std::cerr << "PROGRAM LINK FAILED\n" << info << std::endl;
        }
    }

    void init_font() {
        // Liberation Sans (SIL Open Font License), metrically compatible with Arial but free to
        // redistribute. From the pack when present, otherwise a loose file beside the executable.
        const char* font_file = "LiberationSans-Regular.ttf";
        std::vector<unsigned char> file_buffer;
        size_t packed_size = 0;
        const unsigned char* ttf = assets.find(font_file, &packed_size);
        if (!ttf) {
            std::ifstream in(font_file, std::ios::binary);
            if (!in) {
                std::cerr << "Failed to open " << font_file << ". Ensure assets.dat or the font is beside the executable." << std::endl;
                exit(-1);
            }
            file_buffer.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            if (file_buffer.empty()) {
                std::cerr << "Failed to read " << font_file << std::endl;
                exit(-1);
            }
            ttf = file_buffer.data();
        }
        font_bitmap = new unsigned char[font_bitmap_w * font_bitmap_h];
        int result = stbtt_BakeFontBitmap(ttf, 0, 32.0, font_bitmap, font_bitmap_w, font_bitmap_h, 32, 96, cdata);
        if (result <= 0) {
            std::cerr << "Failed to bake font bitmap. Result: " << result << std::endl;
            delete[] font_bitmap;
            exit(-1);
        }
        glGenTextures(1, &font_tex);
        glBindTexture(GL_TEXTURE_2D, font_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, font_bitmap_w, font_bitmap_h, 0, GL_RED, GL_UNSIGNED_BYTE, font_bitmap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    void init_draw_buffers() {
        glGenVertexArrays(1, &quad_vao);
        glBindVertexArray(quad_vao);
        glGenBuffers(1, &quad_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        glGenVertexArrays(1, &line_vao);
        glBindVertexArray(line_vao);
        glGenBuffers(1, &line_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

        const int segments = 20;
        std::vector<float> circle_verts;
        circle_verts.push_back(0.0f);
        circle_verts.push_back(0.0f);
        circle_verts.push_back(0.0f);
        circle_verts.push_back(0.5f);
        circle_verts.push_back(0.5f);
        for (int i = 0; i <= segments; i++) {
            float theta = 2.0f * glm::pi<float>() * static_cast<float>(i) / static_cast<float>(segments);
            circle_verts.push_back(std::cos(theta));
            circle_verts.push_back(std::sin(theta));
            circle_verts.push_back(0.0f);
            circle_verts.push_back(0.5f + 0.5f * std::cos(theta));
            circle_verts.push_back(0.5f + 0.5f * std::sin(theta));
        }
        glGenVertexArrays(1, &circle_vao);
        glBindVertexArray(circle_vao);
        glGenBuffers(1, &circle_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, circle_vbo);
        glBufferData(GL_ARRAY_BUFFER, circle_verts.size() * sizeof(float), circle_verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }

    void load_textures() {
        stbi_set_flip_vertically_on_load(true);
        wall_tex = load_texture("wall_texture.png");
        boundary_tex = load_texture("boundary_texture.png");
        // Sprites stand on the floor, so measure the empty strip under each figure and drop its
        // billboard by that much. Measured from the image itself, so a replacement texture with
        // different padding still lands on the floor instead of floating.
        monster_tex = load_texture("monster_texture.png", true, nullptr, nullptr, &monster_bottom_margin);
        monster2_tex = load_texture("monster2_texture.png", true, nullptr, nullptr, &boss_bottom_margin);
        medpack_tex = load_texture("medpack.png", true, nullptr, nullptr, &medpack_bottom_margin);

        // The lower level's own wall texture. If it's missing, fall back to the maze wall
        // texture so the lower level is never left untextured.
        lower_wall_tex = load_texture("cavewalltexture.jpg", false);
        if (!lower_wall_tex) {
            std::cerr << "cavewalltexture.jpg not found; using the maze wall texture "
                         "for the lower level." << std::endl;
            lower_wall_tex = wall_tex;
        }

        // The lower level's floor. Its aspect drives the UV tiling below, so a replacement
        // image of any shape still lays down square, undistorted texels.
        int fw = 1, fh = 1;
        lower_floor_tex = load_texture("cavefloorgrok.jpg", false, &fw, &fh);
        if (lower_floor_tex && fh > 0) lower_floor_aspect = static_cast<float>(fw) / fh;
        else std::cerr << "cavefloorgrok.jpg not found; the lower floor will be a flat colour."
                       << std::endl;
    }

    GLuint load_texture(const char* filename, bool warn_if_missing = true,
                        int* out_w = nullptr, int* out_h = nullptr,
                        float* out_bottom_margin = nullptr) {
        int w, h, channels;
        // Prefer the pack; fall back to a loose file so a dev checkout works without one.
        size_t packed_size = 0;
        const unsigned char* packed = assets.find(filename, &packed_size);
        unsigned char* data = packed
            ? stbi_load_from_memory(packed, static_cast<int>(packed_size), &w, &h, &channels, 0)
            : stbi_load(filename, &w, &h, &channels, 0);
        if (!data) {
            if (warn_if_missing) std::cerr << "Failed to load texture " << filename << std::endl;
            return 0;
        }
        if (out_w) *out_w = w;
        if (out_h) *out_h = h;
        if (out_bottom_margin) *out_bottom_margin = transparent_bottom_margin(data, w, h, channels);
        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GLenum format = channels == 4 ? GL_RGBA : GL_RGB;
        glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);
        return tex;
    }

    std::optional<std::vector<std::pair<int, int>>> find_path() {
        std::set<std::pair<int, int>> visited;
        std::queue<std::pair<int, int>> queue;
        std::map<std::pair<int, int>, std::pair<int, int>> parent;
        queue.push(start);
        visited.insert(start);
        parent[start] = start;
        while (!queue.empty()) {
            auto pos = queue.front();
            queue.pop();
            if (pos == end) {
                std::vector<std::pair<int, int>> path;
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

    bool is_solvable() {
        return find_path().has_value();
    }

    void generate_maze() {
        int max_attempts = 10;
        int attempt = 0;
        bool solvable = false;
        while (attempt < max_attempts) {
            grid.assign(grid_size, std::vector<int>(grid_size, 1));
            connections.clear();
            start = { 0, 0 };
            int half_size = grid_size / 2;
            end = { std::uniform_int_distribution<int>(half_size, grid_size - 1)(rng), std::uniform_int_distribution<int>(half_size, grid_size - 1)(rng) };
            std::set<std::pair<int, int>> visited;
            grid[0][0] = 0;
            visited.insert(start);
            std::stack<std::pair<std::pair<int, int>, std::pair<int, int>>> stack;
            stack.push({ start, {0, 0} });

            std::vector<std::pair<int, int>> directions = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
            double extra_branch_prob = 0.3;

            while (!stack.empty()) {
                auto [pos, in_dir] = stack.top();
                int x = pos.first, y = pos.second;
                std::vector<std::tuple<int, int, std::pair<int, int>>> unvisited_neighbors;
                for (const auto& d : directions) {
                    int nx = x + d.first, ny = y + d.second;
                    if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && visited.find({ nx, ny }) == visited.end()) {
                        unvisited_neighbors.emplace_back(nx, ny, d);
                    }
                }
                if (!unvisited_neighbors.empty()) {
                    std::vector<double> weights;
                    double max_dist = grid_size * 2;
                    for (const auto& [nx, ny, d] : unvisited_neighbors) {
                        double dist = std::abs(nx - end.first) + std::abs(ny - end.second);
                        double w = std::pow(max_dist - dist, 2);
                        if (d == in_dir) w *= 5;
                        weights.push_back(std::max(w, 0.001));
                    }
                    size_t chosen_idx = std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng);
                    auto [nx, ny, d] = unvisited_neighbors[chosen_idx];
                    connections[pos].insert({ nx, ny });
                    connections[{nx, ny}].insert(pos);
                    grid[ny][nx] = 0;
                    visited.insert({ nx, ny });
                    stack.push({ {nx, ny}, d });

                    unvisited_neighbors.erase(unvisited_neighbors.begin() + chosen_idx);
                    weights.erase(weights.begin() + chosen_idx);

                    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < extra_branch_prob && !unvisited_neighbors.empty()) {
                        size_t extra_idx = std::uniform_int_distribution<size_t>(0, unvisited_neighbors.size() - 1)(rng);
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
            int num_rooms = std::uniform_int_distribution<int>(1, 3)(rng);
            for (int _ = 0; _ < num_rooms; _++) {
                int room_size = std::uniform_int_distribution<int>(3, 5)(rng);
                int max_start = grid_size - room_size;
                int rx = std::uniform_int_distribution<int>(0, max_start)(rng);
                int ry = std::uniform_int_distribution<int>(0, max_start)(rng);
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

            std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> possible_edges;
            for (int y = 0; y < grid_size; y++) {
                for (int x = 0; x < grid_size; x++) {
                    if (grid[y][x] == 0) {
                        for (const auto& d : directions) {
                            int nx = x + d.first, ny = y + d.second;
                            if (nx > x || (nx == x && ny > y)) {
                                if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && grid[ny][nx] == 0) {
                                    if (connections[{x, y}].find({ nx, ny }) == connections[{x, y}].end()) {
                                        possible_edges.emplace_back(std::make_pair(std::make_pair(x, y), std::make_pair(nx, ny)));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            std::shuffle(possible_edges.begin(), possible_edges.end(), rng);
            int added_loops = 0;
            int max_loops = 10;
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

            if (is_solvable()) {
                solvable = true;
                break;
            }
            attempt++;
        }

        if (!solvable) {
            std::cerr << "Failed to generate solvable maze" << std::endl;
            exit(-1);
        }
    }

    // The lower level is the opposite of the maze: everything is open floor, and only a
    // handful of long straight walls break it into a few huge rooms. Each wall gets a
    // doorway so the whole floor stays walkable.
    void generate_lower_level() {
        lower_grid.assign(grid_size, std::vector<int>(grid_size, 0));
        lower_connections.clear();
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                if (x + 1 < grid_size) link(lower_connections, { x, y }, { x + 1, y });
                if (y + 1 < grid_size) link(lower_connections, { x, y }, { x, y + 1 });
            }
        }

        int num_walls = std::uniform_int_distribution<int>(4, 6)(rng);
        for (int w = 0; w < num_walls; w++) {
            bool vertical = std::uniform_int_distribution<int>(0, 1)(rng) == 0;
            // The grid line the wall sits on, i.e. the seam between cells line-1 and line.
            int line = std::uniform_int_distribution<int>(5, grid_size - 5)(rng);
            int span = std::uniform_int_distribution<int>(12, 22)(rng);
            int span_start = std::uniform_int_distribution<int>(0, grid_size - span)(rng);
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
    void carve_staircase() {
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
                std::pair<int, int> n{ c.first + dxs[d], c.second + dys[d] };
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

    void regenerate_maze() {
        generate_maze();
        generate_lower_level();
        carve_staircase();
        player_level = 0;
        jump_height = 0.0;
        jump_velocity = 0.0;
        player_pos_x = start.first + 0.5;
        player_pos_y = start.second + 0.5;
        spawn_monsters();
        spawn_health_packs();
        projectiles.clear();
        monster_projectiles.clear();
        player_hp = max_hp;
        damage_cooldown = 0;
        build_meshes();
        showing_win = false;
        showing_die = false;
        monster_in_view = false;
        inside_boss_room = -1;
        stop_boss_sound();
    }

    // Reset the player to the maze start without regenerating the maze layout.
    void respawn_player() {
        player_pos_x = start.first + 0.5;
        player_pos_y = start.second + 0.5;
        player_level = 0;
        jump_height = 0.0;
        jump_velocity = 0.0;
        yaw = 0.0;
        pitch = 0.0;
        dir_x = 1.0;
        dir_y = 0.0;
        player_hp = max_hp;
        damage_cooldown = 0;
        projectiles.clear();
        monster_projectiles.clear();
        // Keep the existing monsters/bosses and health packs as they are: anything already
        // killed or collected stays gone. Only the player is reset to the maze start.
        showing_die = false;
        monster_in_view = false;
        inside_boss_room = -1;
        stop_boss_sound();
    }

    // Developer helper: cycle the player through the centers of the maze's boss rooms.
    void dev_teleport_to_boss_room() {
        if (rooms.empty()) return;
        dev_tp_index %= rooms.size();
        int rx, ry, rs;
        std::tie(rx, ry, rs) = rooms[dev_tp_index];
        player_pos_x = (rx + rs / 2) + 0.5;
        player_pos_y = (ry + rs / 2) + 0.5;
        player_level = 0; // boss rooms are all on the maze level
        dev_tp_index = (dev_tp_index + 1) % rooms.size();
    }

    // Developer helper: drop the player right outside the exit (an adjacent, connected cell)
    // so they are next to it but not standing on it.
    void dev_teleport_near_exit() {
        player_level = 0; // the exit is on the maze level
        int ex = end.first, ey = end.second;
        const int dxs[4] = { 1, -1, 0, 0 };
        const int dys[4] = { 0, 0, 1, -1 };
        for (int i = 0; i < 4; ++i) {
            int nx = ex + dxs[i], ny = ey + dys[i];
            if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size) continue;
            if (grid[ny][nx] == 0 && connections[{ex, ey}].count({nx, ny})) {
                player_pos_x = nx + 0.5;
                player_pos_y = ny + 0.5;
                return;
            }
        }
        // Fallback: nearest open cell that isn't the exit itself.
        int best_x = ex, best_y = ey, best_d = 1 << 30;
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                if (grid[y][x] != 0) continue;
                int d = std::abs(x - ex) + std::abs(y - ey);
                if (d > 0 && d < best_d) { best_d = d; best_x = x; best_y = y; }
            }
        }
        player_pos_x = best_x + 0.5;
        player_pos_y = best_y + 0.5;
    }

    void spawn_monsters() {
        monsters.clear();
        std::vector<std::pair<int, int>> path_cells;
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                if (grid[y][x] == 0 && std::pair<int, int>{x, y} != start && std::pair<int, int>{x, y} != end
                    && !is_stair_cell(x, y)) {
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
        std::shuffle(path_cells.begin(), path_cells.end(), rng);
        int num_monsters = std::uniform_int_distribution<int>(5, 12)(rng);
        for (int i = 0; i < std::min(num_monsters, (int)path_cells.size()); i++) {
            int mx = path_cells[i].first, my = path_cells[i].second;
            int hp = 300; // 3 body shots (100 each) or 1 headshot (300) to kill
            monsters.push_back({ mx + 0.5, my + 0.5, hp, 1, mx + 0.5, my + 0.5, std::uniform_int_distribution<int>(0, 180)(rng) });
        }
        for (const auto& room : rooms) {
            int rx, ry, rs;
            std::tie(rx, ry, rs) = room;
            int mx = rx + rs / 2;
            int my = ry + rs / 2;
            int boss_hp = std::uniform_int_distribution<int>(310, 440)(rng); // 7-9 body shots (50 each) or 3 headshots (150) to kill
            monsters.push_back({ mx + 0.5, my + 0.5, boss_hp, 2, mx + 0.5, my + 0.5, std::uniform_int_distribution<int>(0, 60)(rng) });
        }
    }

    void spawn_health_packs() {
        health_packs.clear();
        std::vector<std::pair<int, int>> path_cells;
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                if (grid[y][x] == 0 && std::pair<int, int>{x, y} != start && std::pair<int, int>{x, y} != end
                    && !is_stair_cell(x, y)) path_cells.emplace_back(x, y);
            }
        }
        std::shuffle(path_cells.begin(), path_cells.end(), rng);
        for (int i = 0; i < 2 && i < path_cells.size(); i++) {
            int hx = path_cells[i].first, hy = path_cells[i].second;
            health_packs.push_back({ hx + std::uniform_real_distribution<double>(0.2, 0.8)(rng), hy + std::uniform_real_distribution<double>(0.2, 0.8)(rng), 0 });
        }

        // Two more on the lower level, anywhere on its open floor except the staircase itself.
        std::vector<std::pair<int, int>> lower_cells;
        for (int y = 0; y < grid_size; y++)
            for (int x = 0; x < grid_size; x++)
                if (lower_grid[y][x] == 0 && !is_stair_cell(x, y)) lower_cells.emplace_back(x, y);
        std::shuffle(lower_cells.begin(), lower_cells.end(), rng);
        for (int i = 0; i < 2 && i < static_cast<int>(lower_cells.size()); i++) {
            int hx = lower_cells[i].first, hy = lower_cells[i].second;
            health_packs.push_back({ hx + std::uniform_real_distribution<double>(0.2, 0.8)(rng), hy + std::uniform_real_distribution<double>(0.2, 0.8)(rng), 1 });
        }
    }

    bool try_move(double new_x, double new_y) {
        int new_cell_x = static_cast<int>(new_x);
        int new_cell_y = static_cast<int>(new_y);
        if (new_cell_x < 0 || new_cell_x >= grid_size || new_cell_y < 0 || new_cell_y >= grid_size) return false;
        if (grid_for(player_level)[new_cell_y][new_cell_x] != 0) return false;
        int old_cell_x = static_cast<int>(player_pos_x);
        int old_cell_y = static_cast<int>(player_pos_y);
        if (new_cell_x == old_cell_x && new_cell_y == old_cell_y) return true;
        return conn_for(player_level)[{old_cell_x, old_cell_y}].count({ new_cell_x, new_cell_y }) > 0;
    }

    double clip_position(double new_pos, double current_pos, bool is_x) {
        // grid is indexed grid[y][x]; connections use (x, y) keys
        // A wall exists on a side whenever we cannot pass to that neighbor cell (out of bounds,
        // solid, or simply not connected in the maze). Keep the player min_dist away from it.
        auto& conn = conn_for(player_level);
        if (is_x) {
            int cell_x = static_cast<int>(current_pos);
            int cell_y = static_cast<int>(player_pos_y);
            if (new_pos > current_pos) {
                bool has_wall = cell_x >= grid_size - 1
                    || conn[{cell_x, cell_y}].count({cell_x + 1, cell_y}) == 0;
                if (has_wall) new_pos = std::min(new_pos, static_cast<double>(cell_x) + 1.0 - min_dist);
            } else {
                bool has_wall = cell_x <= 0
                    || conn[{cell_x, cell_y}].count({cell_x - 1, cell_y}) == 0;
                if (has_wall) new_pos = std::max(new_pos, static_cast<double>(cell_x) + min_dist);
            }
        } else {
            int cell_x = static_cast<int>(player_pos_x);
            int cell_y = static_cast<int>(current_pos);
            if (new_pos > current_pos) {
                bool has_wall = cell_y >= grid_size - 1
                    || conn[{cell_x, cell_y}].count({cell_x, cell_y + 1}) == 0;
                if (has_wall) new_pos = std::min(new_pos, static_cast<double>(cell_y) + 1.0 - min_dist);
            } else {
                bool has_wall = cell_y <= 0
                    || conn[{cell_x, cell_y}].count({cell_x, cell_y - 1}) == 0;
                if (has_wall) new_pos = std::max(new_pos, static_cast<double>(cell_y) + min_dist);
            }
        }
        return new_pos;
    }

    void shoot() {
        // Slow enough to read as a travelling bolt. Maze sightlines are short - at the old
        // 0.28 a shot reached the wall in ~0.09s, well under the fire interval, so there was
        // never more than one in the air.
        double speed = 0.12;
        // Aim vector combines horizontal facing (yaw) with vertical pitch.
        double cp = std::cos(glm::radians(pitch));
        double sp = std::sin(glm::radians(pitch));
        double aim_x = dir_x * cp;
        double aim_y = dir_y * cp;
        double aim_z = sp;
        double start_x = player_pos_x + aim_x * 0.6;
        double start_y = player_pos_y + aim_y * 0.6;
        double start_z = player_eye_y() + aim_z * 0.6;
        Projectile p;
        p.x = start_x; p.y = start_y; p.z = start_z;
        p.dir_x = aim_x; p.dir_y = aim_y; p.dir_z = aim_z;
        p.speed = speed;
        p.level = player_level;
        projectiles.push_back(p);
    }

    void update_projectiles(double delta) {
        for (auto it = projectiles.begin(); it != projectiles.end(); ) {
            double move_dist = it->speed * delta * 60.0;
            const double step_len = 0.08;
            int steps = std::max(1, static_cast<int>(std::ceil(move_dist / step_len)));
            double sx = (it->dir_x * move_dist) / steps;
            double sy = (it->dir_y * move_dist) / steps;
            double sz = (it->dir_z * move_dist) / steps;
            bool hit = false;
            bool headshot = false;
            Monster* hit_monster = nullptr;
            // Each level has its own walls and its own floor/ceiling heights.
            const auto& lvl_grid = grid_for(it->level);
            auto& lvl_conn = conn_for(it->level);
            double z_floor = it->level == 0 ? 0.0 : lower_floor_y;
            double z_ceil = it->level == 0 ? 1.0 : 0.0;
            for (int s = 0; s < steps; ++s) {
                double new_x = it->x + sx;
                double new_y = it->y + sy;
                double new_z = it->z + sz;
                int old_cell_x = static_cast<int>(it->x);
                int old_cell_y = static_cast<int>(it->y);
                int new_cell_x = static_cast<int>(new_x);
                int new_cell_y = static_cast<int>(new_y);
                // Meeting the floor or ceiling flattens the shot out rather than killing it.
                // The maze is only one unit tall, so a shot fired with any real pitch reaches
                // them within a few frames - despawning there made shots vanish in front of
                // you instead of carrying on to a wall.
                if (new_z < z_floor) { new_z = z_floor; it->dir_z = 0.0; sz = 0.0; }
                else if (new_z > z_ceil) { new_z = z_ceil; it->dir_z = 0.0; sz = 0.0; }

                bool hit_wall = false;
                if (new_cell_x < 0 || new_cell_x >= grid_size || new_cell_y < 0 || new_cell_y >= grid_size) hit_wall = true;
                else if (lvl_grid[new_cell_y][new_cell_x] != 0) hit_wall = true;
                else if (new_cell_x != old_cell_x || new_cell_y != old_cell_y) {
                    if (std::abs(new_cell_x - old_cell_x) + std::abs(new_cell_y - old_cell_y) > 1) hit_wall = true;
                    else if (lvl_conn[{old_cell_x, old_cell_y}].find({new_cell_x, new_cell_y}) == lvl_conn[{old_cell_x, old_cell_y}].end()) hit_wall = true;
                }
                // Monsters live on the maze level and span height 0..1; only register a hit
                // when the shot is on their level and at their height.
                if (it->level == 0) {
                    for (auto& m : monsters) {
                        // Height within the monster's billboard, which is sunk to put its feet
                        // on the floor - the hit volume moves with what is drawn.
                        double z_on_sprite = new_z + monster_sink(m.type);
                        if (std::hypot(m.x - new_x, m.y - new_y) < 0.5 && z_on_sprite >= 0.0 && z_on_sprite <= 1.0) {
                            hit_monster = &m;
                            headshot = (z_on_sprite >= 0.75); // top quarter is the head
                            break;
                        }
                    }
                }
                if (hit_wall || hit_monster) { hit = true; break; }
                it->x = new_x; it->y = new_y; it->z = new_z;
            }
            if (hit) {
                it = projectiles.erase(it);
                if (hit_monster) {
                    int dmg;
                    if (hit_monster->type == 1)
                        dmg = headshot ? 300 : 100; // 1 headshot or 3 body shots
                    else
                        dmg = headshot ? 150 : 50;  // 3 headshots or 7-9 body shots
                    hit_monster->hp -= dmg;
                    hit_monster->hit_flash = 8; // briefly highlight red on a successful hit
                    if (hit_monster->hp <= 0)
                        monsters.erase(std::remove_if(monsters.begin(), monsters.end(), [hit_monster](const Monster& m){return &m==hit_monster;}), monsters.end());
                }
            } else ++it;
        }
    }

    void update_monster_projectiles(double delta) {
        for (auto it = monster_projectiles.begin(); it != monster_projectiles.end(); ) {
            double move_dist = it->speed * delta * 60.0;
            const double step_len = 0.08;
            int steps = std::max(1, static_cast<int>(std::ceil(move_dist / step_len)));
            double sx = (it->dir_x * move_dist) / steps;
            double sy = (it->dir_y * move_dist) / steps;
            bool hit = false, hit_player = false;
            for (int s = 0; s < steps; ++s) {
                double new_x = it->x + sx;
                double new_y = it->y + sy;
                int old_cell_x = static_cast<int>(it->x);
                int old_cell_y = static_cast<int>(it->y);
                int new_cell_x = static_cast<int>(new_x);
                int new_cell_y = static_cast<int>(new_y);
                bool hit_wall = false;
                if (new_cell_x < 0 || new_cell_x >= grid_size || new_cell_y < 0 || new_cell_y >= grid_size) hit_wall = true;
                else if (grid[new_cell_y][new_cell_x] != 0) hit_wall = true;
                else if (new_cell_x != old_cell_x || new_cell_y != old_cell_y) {
                    if (std::abs(new_cell_x - old_cell_x) + std::abs(new_cell_y - old_cell_y) > 1) hit_wall = true;
                    else if (connections[{old_cell_x, old_cell_y}].find({new_cell_x, new_cell_y}) == connections[{old_cell_x, old_cell_y}].end()) hit_wall = true;
                }
                // Monster shots travel the maze level only; the player is out of reach downstairs.
                if (player_level == 0 && std::hypot(new_x - player_pos_x, new_y - player_pos_y) < 0.4) hit_player = true;
                if (hit_wall || hit_player) { hit = true; break; }
                it->x = new_x; it->y = new_y;
            }
            if (hit) {
                bool was_boss = it->from_boss;
                it = monster_projectiles.erase(it);
                if (hit_player) player_hp -= (was_boss ? std::uniform_int_distribution<int>(4, 5)(rng) : 1); // 4-5 boss shots kill
            } else ++it;
        }
    }

    void update_monsters(double delta) {
        std::vector<std::pair<int, int>> directions = { {-1, 0}, {1, 0}, {0, -1}, {0, 1} };
        for (auto it = monsters.begin(); it != monsters.end(); ) {
            if (it->hit_flash > 0) it->hit_flash--; // fade out the red hit highlight
            double dx = it->target_x - it->x;
            double dy = it->target_y - it->y;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist < 0.01) {
                int cx = static_cast<int>(it->x);
                int cy = static_cast<int>(it->y);
                std::vector<std::pair<double, double>> possible_targets;
                for (const auto& d : directions) {
                    int nx = cx + d.first, ny = cy + d.second;
                    // Never wander onto the staircase: monsters have no notion of height and
                    // would float above the steps.
                    if (nx >= 0 && nx < grid_size && ny >= 0 && ny < grid_size && grid[ny][nx] == 0
                        && connections[{cx, cy}].count({ nx, ny }) && !is_stair_cell(nx, ny)) {
                        possible_targets.emplace_back(nx + 0.5, ny + 0.5);
                    }
                }
                if (!possible_targets.empty()) {
                    auto target = possible_targets[std::uniform_int_distribution<size_t>(0, possible_targets.size() - 1)(rng)];
                    it->target_x = target.first;
                    it->target_y = target.second;
                }
            }
            else {
                double speed = (it->type == 1 ? 0.0072 : 0.01275) * delta * 60.0; // monster -28%, boss -15%
                it->x += (dx / dist) * speed;
                it->y += (dy / dist) * speed;
            }

            it->cooldown -= 1;
            if (it->cooldown <= 0) {
                double dx_p = player_pos_x - it->x;
                double dy_p = player_pos_y - it->y;
                double p_dist = std::hypot(dx_p, dy_p);
                // Hold fire while the player is on the lower level - the shot could not reach.
                // The cooldown still resets below so they don't all volley the moment you return.
                if (p_dist > 0 && player_level == 0) {
                    double dir_x_p = dx_p / p_dist;
                    double dir_y_p = dy_p / p_dist;
                    bool is_boss = (it->type == 2);
                    Projectile mp;
                    mp.x = it->x; mp.y = it->y;
                    mp.dir_x = dir_x_p; mp.dir_y = dir_y_p;
                    mp.speed = is_boss ? 0.0714 : 0.075; // boss projectiles slower than regular ones
                    mp.from_boss = is_boss;
                    monster_projectiles.push_back(mp);
                }
                it->cooldown = (it->type == 2 ? 86 : 300); // boss fires 30% less often, monsters 40% less
            }

            ++it;
        }
    }

    // Emit the wall quads implied by a level's grid + connection map: a wall stands on any cell
    // edge the player cannot cross. Walls span [y0, y1] in world space and the texture tiles
    // vertically over that height, so the taller lower level doesn't look stretched.
    void build_level_walls(const std::vector<std::vector<int>>& g,
                           std::map<std::pair<int, int>, std::set<std::pair<int, int>>>& conn,
                           float y0, float y1,
                           std::vector<float>& walls, std::vector<unsigned int>& wall_inds,
                           std::vector<float>& bounds, std::vector<unsigned int>& bound_inds,
                           std::vector<float>& mini,
                           float uv_scale = 1.0f) {
        // UVs come from world position rather than 0..1 per quad, so one texture tile can span
        // several cells and neighbouring segments line up instead of each restarting the image.
        // At uv_scale 1 that is identical to a per-cell 0..1 mapping, since the texture repeats.
        const float s = uv_scale;
        const float v_top = (y1 - y0) / s;
        const unsigned int quad_inds[] = { 0, 1, 2, 0, 2, 3 };

        auto emit = [&](const float* verts, bool is_boundary) {
            if (is_boundary) {
                unsigned int base = static_cast<unsigned int>(bounds.size() / 5);
                bounds.insert(bounds.end(), verts, verts + 20);
                for (auto i : quad_inds) bound_inds.push_back(base + i);
            }
            else {
                unsigned int base = static_cast<unsigned int>(walls.size() / 5);
                walls.insert(walls.end(), verts, verts + 20);
                for (auto i : quad_inds) wall_inds.push_back(base + i);
            }
        };

        for (int x = 0; x <= grid_size; x++) {
            for (int z = 0; z < grid_size; z++) {
                bool draw = (x == 0 || x == grid_size);
                if (!draw && x > 0) {
                    bool both_path = g[z][x - 1] == 0 && g[z][x] == 0;
                    bool connected = conn[{x - 1, z}].count({ x, z });
                    draw = !(both_path && connected);
                }
                if (draw) {
                    float u0 = z / s, u1 = (z + 1) / s;
                    float verts[] = {
                        static_cast<float>(x), y0, static_cast<float>(z), u0, 0.0f,
                        static_cast<float>(x), y0, static_cast<float>(z + 1), u1, 0.0f,
                        static_cast<float>(x), y1, static_cast<float>(z + 1), u1, v_top,
                        static_cast<float>(x), y1, static_cast<float>(z), u0, v_top
                    };
                    emit(verts, x == 0 || x == grid_size);
                    mini.push_back(static_cast<float>(x));
                    mini.push_back(static_cast<float>(z));
                    mini.push_back(0.0f);
                    mini.push_back(static_cast<float>(x));
                    mini.push_back(static_cast<float>(z + 1));
                    mini.push_back(0.0f);
                }
            }
        }

        for (int z = 0; z <= grid_size; z++) {
            for (int x = 0; x < grid_size; x++) {
                bool draw = (z == 0 || z == grid_size);
                if (!draw && z > 0) {
                    bool both_path = g[z - 1][x] == 0 && g[z][x] == 0;
                    bool connected = conn[{x, z - 1}].count({ x, z });
                    draw = !(both_path && connected);
                }
                if (draw) {
                    float u0 = x / s, u1 = (x + 1) / s;
                    float verts[] = {
                        static_cast<float>(x), y0, static_cast<float>(z), u0, 0.0f,
                        static_cast<float>(x + 1), y0, static_cast<float>(z), u1, 0.0f,
                        static_cast<float>(x + 1), y1, static_cast<float>(z), u1, v_top,
                        static_cast<float>(x), y1, static_cast<float>(z), u0, v_top
                    };
                    emit(verts, z == 0 || z == grid_size);
                    mini.push_back(static_cast<float>(x));
                    mini.push_back(static_cast<float>(z));
                    mini.push_back(0.0f);
                    mini.push_back(static_cast<float>(x + 1));
                    mini.push_back(static_cast<float>(z));
                    mini.push_back(0.0f);
                }
            }
        }
    }

    // The staircase: a run of treads and risers dropping from the maze floor to the lower one.
    // The shaft's side walls come for free from the two levels' wall meshes, since the run is
    // sealed off from its neighbours in both connection maps.
    void build_stair_mesh(std::vector<float>& verts, std::vector<unsigned int>& inds) {
        verts.clear();
        inds.clear();
        if (stair_cells.empty()) return;

        const int steps = 20;
        const float x0 = static_cast<float>(stair_cells.front().first);
        const float z0 = static_cast<float>(stair_cells.front().second);
        const float run = static_cast<float>(stair_cells.size()) / steps;

        auto quad = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, float u1, float v1) {
            unsigned int base = static_cast<unsigned int>(verts.size() / 5);
            float q[] = {
                a.x, a.y, a.z, 0.0f, 0.0f,
                b.x, b.y, b.z, u1,   0.0f,
                c.x, c.y, c.z, u1,   v1,
                d.x, d.y, d.z, 0.0f, v1
            };
            verts.insert(verts.end(), q, q + 20);
            for (unsigned int i : { 0u, 1u, 2u, 0u, 2u, 3u }) inds.push_back(base + i);
        };

        auto tread_y = [&](int k) {
            return static_cast<float>(lower_floor_y * ((k + 0.5) / steps));
        };

        for (int k = 0; k < steps; k++) {
            float y = tread_y(k);
            float xa = x0 + k * run, xb = x0 + (k + 1) * run;
            // Tread: the surface you walk on.
            quad({ xa, y, z0 }, { xb, y, z0 }, { xb, y, z0 + 1 }, { xa, y, z0 + 1 }, run, 1.0f);
            // Riser: the vertical face at the back of the tread, i.e. what you see looking down.
            if (k > 0) {
                float prev = tread_y(k - 1);
                quad({ xa, y, z0 }, { xa, y, z0 + 1 }, { xa, prev, z0 + 1 }, { xa, prev, z0 },
                     1.0f, prev - y);
            }
        }

        // Final lip down onto the lower floor.
        float xn = x0 + static_cast<float>(stair_cells.size());
        float last = tread_y(steps - 1);
        quad({ xn, static_cast<float>(lower_floor_y), z0 }, { xn, static_cast<float>(lower_floor_y), z0 + 1 },
             { xn, last, z0 + 1 }, { xn, last, z0 }, 1.0f, last - static_cast<float>(lower_floor_y));
    }

    void upload_gun(const std::vector<float>& verts, const std::vector<unsigned int>& inds) {
        glGenVertexArrays(1, &gun_vao);
        glBindVertexArray(gun_vao);
        glGenBuffers(1, &gun_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, gun_vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
        glGenBuffers(1, &gun_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gun_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.size() * sizeof(unsigned int), inds.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }

    // The player's weapon: a blocky sci-fi carbine assembled from boxes. Its palette is taken
    // from Sci-Fi_Gun_Texture.png and Sci-Fi_Gun_Emission.png - matte near-black shell with
    // cyan trim. Built once at startup; it never changes.
    void build_gun_mesh() {
        const glm::vec3 BODY(0.16f, 0.20f, 0.21f); // shell
        const glm::vec3 DARK(0.07f, 0.09f, 0.10f); // grip, magazine, sights
        const glm::vec3 TRIM(0.33f, 0.97f, 0.97f); // cyan from the albedo
        const glm::vec3 GLOW(0.66f, 0.99f, 0.99f); // brighter cyan from the emission map

        struct Face { glm::vec3 v[4]; glm::vec3 n; glm::vec3 col; bool emissive; };
        std::vector<Face> faces;

        auto add_box = [&](glm::vec3 c, glm::vec3 h, glm::mat3 rot, glm::vec3 col, bool emissive) {
            const glm::vec3 nrm[6] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
            const int idx[6][4] = { {1,5,7,3}, {4,0,2,6}, {2,3,7,6}, {4,5,1,0}, {5,4,6,7}, {0,1,3,2} };
            glm::vec3 p[8];
            for (int i = 0; i < 8; i++) {
                glm::vec3 local((i & 1) ? h.x : -h.x, (i & 2) ? h.y : -h.y, (i & 4) ? h.z : -h.z);
                p[i] = c + rot * local;
            }
            for (int fi = 0; fi < 6; fi++) {
                Face face;
                for (int k = 0; k < 4; k++) face.v[k] = p[idx[fi][k]];
                face.n = rot * nrm[fi];
                face.col = col;
                face.emissive = emissive;
                faces.push_back(face);
            }
        };
        auto rotx = [](float deg) {
            float r = glm::radians(deg), c = std::cos(r), s = std::sin(r);
            return glm::mat3(1, 0, 0, 0, c, s, 0, -s, c);
        };
        const glm::mat3 I(1.0f);

        // Forward is -Z. The gun is seen from behind and slightly above, so the barrel rides
        // above the receiver's top line - otherwise the receiver hides it and this reads as a
        // featureless block.
        add_box({ 0.000f, -0.015f, -0.140f }, { 0.040f, 0.040f, 0.165f }, I, BODY, false); // receiver
        add_box({ 0.000f,  0.040f, -0.300f }, { 0.030f, 0.026f, 0.300f }, I, BODY, false); // barrel
        add_box({ 0.000f,  0.040f, -0.630f }, { 0.038f, 0.034f, 0.035f }, I, DARK, false); // muzzle brake
        add_box({ 0.000f,  0.078f, -0.060f }, { 0.020f, 0.014f, 0.050f }, I, DARK, false); // rear sight
        add_box({ 0.000f,  0.082f, -0.500f }, { 0.008f, 0.018f, 0.012f }, I, DARK, false); // front post
        add_box({ 0.000f, -0.100f, -0.120f }, { 0.026f, 0.070f, 0.038f }, rotx(-6.0f), DARK, false); // magazine
        add_box({ 0.000f, -0.100f,  0.030f }, { 0.028f, 0.070f, 0.026f }, rotx(18.0f), DARK, false); // grip
        add_box({ 0.000f, -0.010f,  0.060f }, { 0.032f, 0.042f, 0.035f }, I, BODY, false); // stock

        add_box({ 0.031f, 0.045f, -0.320f }, { 0.003f, 0.012f, 0.240f }, I, GLOW, true); // flank strips
        add_box({ -0.031f, 0.045f, -0.320f }, { 0.003f, 0.012f, 0.240f }, I, GLOW, true);
        add_box({ 0.000f, 0.067f, -0.340f }, { 0.009f, 0.003f, 0.190f }, I, TRIM, true);  // spine band
        add_box({ 0.000f, 0.040f, -0.598f }, { 0.033f, 0.029f, 0.006f }, I, GLOW, true);  // muzzle ring

        // Bake a key light into the face colours, then group faces by the colour that produces
        // so the whole gun draws in a handful of calls.
        const glm::vec3 light = glm::normalize(glm::vec3(-0.35f, 0.85f, 0.40f));
        auto key_of = [](glm::vec3 c) {
            auto q = [](float v) { return static_cast<unsigned int>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
            return (q(c.r) << 16) | (q(c.g) << 8) | q(c.b);
        };

        std::vector<float> verts;
        std::map<unsigned int, std::vector<unsigned int>> by_color;
        for (const auto& f : faces) {
            float shade = f.emissive ? 1.0f
                : 0.30f + 0.70f * std::max(0.0f, glm::dot(glm::normalize(f.n), light));
            unsigned int base = static_cast<unsigned int>(verts.size() / 5);
            for (int k = 0; k < 4; k++) {
                verts.push_back(f.v[k].x); verts.push_back(f.v[k].y); verts.push_back(f.v[k].z);
                verts.push_back(0.0f); verts.push_back(0.0f); // unlit and untextured
            }
            auto& group = by_color[key_of(f.col * shade)];
            for (unsigned int i : { 0u, 1u, 2u, 0u, 2u, 3u }) group.push_back(base + i);
        }

        std::vector<unsigned int> inds;
        gun_sections.clear();
        for (const auto& [k, list] : by_color) {
            gun_sections.push_back({ inds.size(), list.size(),
                glm::vec4(((k >> 16) & 255) / 255.0f, ((k >> 8) & 255) / 255.0f, (k & 255) / 255.0f, 1.0f) });
            inds.insert(inds.end(), list.begin(), list.end());
        }
        upload_gun(verts, inds);
    }

    // Drawn in its own little scene pinned to the camera, so level geometry never clips it.
    void render_viewmodel() {
        glViewport(0, 0, screen_width, screen_height);
        glClear(GL_DEPTH_BUFFER_BIT); // the gun sits in front of everything...
        glEnable(GL_DEPTH_TEST);      // ...but still self-occludes correctly

        glm::mat4 projection = glm::perspective(glm::radians(60.0f),
            static_cast<float>(screen_width) / screen_height, 0.01f, 10.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glm::mat4 view = glm::mat4(1.0f); // camera at the origin looking down -Z
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));

        // recoil counts down from recoil_time after each shot, giving a kick envelope.
        // Squared so the kick snaps and then settles.
        float kick = static_cast<float>(std::max(0.0, recoil) / recoil_time);
        kick *= kick;

        // Held right of centre and low, angled so the barrel converges toward the crosshair.
        glm::mat4 model = glm::translate(glm::mat4(1.0f),
            glm::vec3(0.22f, -0.17f + 0.012f * kick, -0.62f + 0.050f * kick));
        model = glm::rotate(model, glm::radians(-11.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(2.0f + 7.0f * kick), glm::vec3(1.0f, 0.0f, 0.0f));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));

        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
        glBindVertexArray(gun_vao);
        for (const auto& s : gun_sections) {
            glUniform4f(glGetUniformLocation(shader_program, "color"), s.color.x, s.color.y, s.color.z, s.color.w);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(s.count), GL_UNSIGNED_INT,
                           (void*)(s.offset * sizeof(unsigned int)));
        }

        glm::mat4 id = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(id));
    }

    void build_meshes() {
        wall_vertices.clear();
        boundary_vertices.clear();
        mini_wall_verts.clear();
        mini_lower_verts.clear();
        std::vector<unsigned int> wall_indices, boundary_indices;

        build_level_walls(grid, connections, 0.0f, 1.0f,
                          wall_vertices, wall_indices, boundary_vertices, boundary_indices,
                          mini_wall_verts);

        wall_index_count = wall_indices.size();
        boundary_index_count = boundary_indices.size();

        glGenVertexArrays(1, &wall_vao);
        glBindVertexArray(wall_vao);
        glGenBuffers(1, &wall_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, wall_vbo);
        glBufferData(GL_ARRAY_BUFFER, wall_vertices.size() * sizeof(float), wall_vertices.data(), GL_STATIC_DRAW);
        glGenBuffers(1, &wall_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, wall_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, wall_indices.size() * sizeof(unsigned int), wall_indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        glGenVertexArrays(1, &boundary_vao);
        glBindVertexArray(boundary_vao);
        glGenBuffers(1, &boundary_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, boundary_vbo);
        glBufferData(GL_ARRAY_BUFFER, boundary_vertices.size() * sizeof(float), boundary_vertices.data(), GL_STATIC_DRAW);
        glGenBuffers(1, &boundary_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boundary_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, boundary_indices.size() * sizeof(unsigned int), boundary_indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        // Upper floor, one quad per cell so the staircase can leave an open shaft through it.
        // Seen from below it doubles as the lower level's ceiling.
        std::vector<float> floor_verts;
        std::vector<unsigned int> floor_inds;
        for (int z = 0; z < grid_size; z++) {
            for (int x = 0; x < grid_size; x++) {
                if (is_stair_cell(x, z)) continue;
                unsigned int base = static_cast<unsigned int>(floor_verts.size() / 5);
                float fx = static_cast<float>(x), fz = static_cast<float>(z);
                float v[] = {
                    fx,        0.0f, fz,        0.0f, 0.0f,
                    fx + 1.0f, 0.0f, fz,        1.0f, 0.0f,
                    fx + 1.0f, 0.0f, fz + 1.0f, 1.0f, 1.0f,
                    fx,        0.0f, fz + 1.0f, 0.0f, 1.0f
                };
                floor_verts.insert(floor_verts.end(), v, v + 20);
                for (unsigned int i : { 0u, 1u, 2u, 0u, 2u, 3u }) floor_inds.push_back(base + i);
            }
        }
        floor_index_count = floor_inds.size();
        glGenVertexArrays(1, &floor_vao);
        glBindVertexArray(floor_vao);
        glGenBuffers(1, &floor_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, floor_vbo);
        glBufferData(GL_ARRAY_BUFFER, floor_verts.size() * sizeof(float), floor_verts.data(), GL_STATIC_DRAW);
        glGenBuffers(1, &floor_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, floor_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, floor_inds.size() * sizeof(unsigned int), floor_inds.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        float ceiling_verts[] = {
            0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
            static_cast<float>(grid_size), 1.0f, 0.0f, 1.0f, 0.0f,
            static_cast<float>(grid_size), 1.0f, static_cast<float>(grid_size), 1.0f, 1.0f,
            0.0f, 1.0f, static_cast<float>(grid_size), 0.0f, 1.0f
        };
        unsigned int ceiling_inds[] = { 0, 1, 2, 0, 2, 3 };
        glGenVertexArrays(1, &ceiling_vao);
        glBindVertexArray(ceiling_vao);
        glGenBuffers(1, &ceiling_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, ceiling_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(ceiling_verts), ceiling_verts, GL_STATIC_DRAW);
        glGenBuffers(1, &ceiling_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ceiling_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ceiling_inds), ceiling_inds, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        mini_wall_vertex_count = mini_wall_verts.size() / 3;
        glGenVertexArrays(1, &mini_wall_vao);
        glBindVertexArray(mini_wall_vao);
        glGenBuffers(1, &mini_wall_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, mini_wall_vbo);
        glBufferData(GL_ARRAY_BUFFER, mini_wall_verts.size() * sizeof(float), mini_wall_verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

        // --- Lower level ------------------------------------------------------------------
        // Its walls run from the lower floor up to the underside of the maze floor, which is
        // twice the maze's wall height and is what makes the rooms down there feel cavernous.
        std::vector<float> lower_wall_verts, lower_boundary_verts;
        std::vector<unsigned int> lower_wall_indices, lower_boundary_indices;
        // One tile of rock every 4 units, so the photo's vignette doesn't read as a grid.
        build_level_walls(lower_grid, lower_connections, static_cast<float>(lower_floor_y), 0.0f,
                          lower_wall_verts, lower_wall_indices,
                          lower_boundary_verts, lower_boundary_indices, mini_lower_verts, 4.0f);
        lower_wall_index_count = lower_wall_indices.size();
        lower_boundary_index_count = lower_boundary_indices.size();

        glGenVertexArrays(1, &lower_wall_vao);
        glBindVertexArray(lower_wall_vao);
        glGenBuffers(1, &lower_wall_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, lower_wall_vbo);
        glBufferData(GL_ARRAY_BUFFER, lower_wall_verts.size() * sizeof(float), lower_wall_verts.data(), GL_STATIC_DRAW);
        glGenBuffers(1, &lower_wall_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lower_wall_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, lower_wall_indices.size() * sizeof(unsigned int), lower_wall_indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        glGenVertexArrays(1, &lower_boundary_vao);
        glBindVertexArray(lower_boundary_vao);
        glGenBuffers(1, &lower_boundary_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, lower_boundary_vbo);
        glBufferData(GL_ARRAY_BUFFER, lower_boundary_verts.size() * sizeof(float), lower_boundary_verts.data(), GL_STATIC_DRAW);
        glGenBuffers(1, &lower_boundary_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lower_boundary_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, lower_boundary_indices.size() * sizeof(unsigned int), lower_boundary_indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        // The floor slab tiles its texture rather than stretching one copy over the whole map.
        // The v span is divided by the image's aspect so the texels stay square whatever shape
        // the source image is.
        float lf = static_cast<float>(lower_floor_y);
        float gs = static_cast<float>(grid_size);
        const float floor_tile_u = 4.0f; // world units per tile across X
        const float floor_tile_v = floor_tile_u / lower_floor_aspect;
        float fu = gs / floor_tile_u;
        float fv = gs / floor_tile_v;
        float lower_floor_verts[] = {
            0.0f, lf, 0.0f, 0.0f, 0.0f,
            gs,   lf, 0.0f, fu,   0.0f,
            gs,   lf, gs,   fu,   fv,
            0.0f, lf, gs,   0.0f, fv
        };
        unsigned int lower_floor_inds[] = { 0, 1, 2, 0, 2, 3 };
        glGenVertexArrays(1, &lower_floor_vao);
        glBindVertexArray(lower_floor_vao);
        glGenBuffers(1, &lower_floor_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, lower_floor_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(lower_floor_verts), lower_floor_verts, GL_STATIC_DRAW);
        glGenBuffers(1, &lower_floor_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lower_floor_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(lower_floor_inds), lower_floor_inds, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        mini_lower_vertex_count = mini_lower_verts.size() / 3;
        glGenVertexArrays(1, &mini_lower_vao);
        glBindVertexArray(mini_lower_vao);
        glGenBuffers(1, &mini_lower_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, mini_lower_vbo);
        glBufferData(GL_ARRAY_BUFFER, mini_lower_verts.size() * sizeof(float), mini_lower_verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

        std::vector<float> stair_verts;
        std::vector<unsigned int> stair_indices;
        build_stair_mesh(stair_verts, stair_indices);
        stair_index_count = stair_indices.size();
        glGenVertexArrays(1, &stair_vao);
        glBindVertexArray(stair_vao);
        glGenBuffers(1, &stair_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, stair_vbo);
        glBufferData(GL_ARRAY_BUFFER, stair_verts.size() * sizeof(float), stair_verts.data(), GL_STATIC_DRAW);
        glGenBuffers(1, &stair_ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, stair_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, stair_indices.size() * sizeof(unsigned int), stair_indices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }

    void run() {
        last_time = glfwGetTime();
        double last_mouse_x = screen_width / 2.0;
        double last_mouse_y = screen_height / 2.0;
        while (!glfwWindowShouldClose(window)) {
            double current_time = glfwGetTime();
            double delta = current_time - last_time;
            last_time = current_time;

            process_input(delta, last_mouse_x, last_mouse_y);

            update_projectiles(delta);
            update_monster_projectiles(delta);
            update_monsters(delta);
            update_audio_cues();

            // Monsters and health packs sit on the maze level; standing under them doesn't count.
            if (damage_cooldown > 0) damage_cooldown--;
            else if (player_level == 0) {
                for (auto& m : monsters) {
                    double dist = std::hypot(m.x - player_pos_x, m.y - player_pos_y);
                    if (dist < 0.8) {
                        player_hp -= 1;
                        damage_cooldown = 30;
                        break;
                    }
                }
            }

            for (auto it = health_packs.begin(); it != health_packs.end(); ) {
                if (it->level == player_level && std::hypot(it->x - player_pos_x, it->y - player_pos_y) < MEDPACK_PICKUP_RADIUS) {
                    player_hp = max_hp;
                    it = health_packs.erase(it);
                }
                else {
                    ++it;
                }
            }

            int boss_count = 0;
            for (const auto& m : monsters) if (m.type == 2) boss_count++;
            if (!showing_win && player_level == 0 && static_cast<int>(player_pos_x) == end.first && static_cast<int>(player_pos_y) == end.second && boss_count == 0) {
                showing_win = true;
                win_timer = current_time + 1.0;
            }

            if (!showing_die && player_hp <= 0) {
                showing_die = true;
                die_timer = current_time + 1.5;
            }

            if (showing_win && current_time > win_timer) {
                regenerate_maze();
            }
            else if (showing_die && current_time > die_timer) {
                respawn_player(); // same maze, back to the start
            }

            glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glUseProgram(shader_program);

            if (tab_view) {
                glDisable(GL_DEPTH_TEST);
                render_2d();
                glEnable(GL_DEPTH_TEST);
            }
            else {
                render_3d();
                render_viewmodel();
                render_minimap();
            }

            render_hud(boss_count);

            if (showing_win) {
                glDisable(GL_DEPTH_TEST);
                draw_text("You Won!", static_cast<float>(screen_width / 2 - 100), static_cast<float>(screen_height / 2), { 1,1,0,1 });
                glEnable(GL_DEPTH_TEST);
            }
            else if (showing_die) {
                glDisable(GL_DEPTH_TEST);
                draw_text("You Died!", static_cast<float>(screen_width / 2 - 100), static_cast<float>(screen_height / 2), { 1,0,0,1 });
                glEnable(GL_DEPTH_TEST);
            }
            else {
                // Standing on the exit with bosses still alive: tell the player it's locked.
                bool at_exit = player_level == 0 && static_cast<int>(player_pos_x) == end.first && static_cast<int>(player_pos_y) == end.second;
                if (at_exit && boss_count > 0) {
                    glDisable(GL_DEPTH_TEST);
                    std::string msg = "Exit locked: " + std::to_string(boss_count)
                        + (boss_count == 1 ? " boss remaining" : " bosses remaining");
                    draw_text(msg, static_cast<float>(screen_width / 2 - 220), static_cast<float>(screen_height / 2), { 1.0f, 0.3f, 0.3f, 1.0f });
                    glEnable(GL_DEPTH_TEST);
                }
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void process_input(double delta, double& last_mouse_x, double& last_mouse_y) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

        if (glfwGetKey(window, GLFW_KEY_F8) == GLFW_PRESS) regenerate_maze();

        // Tab toggles the map view (press to switch, no longer hold-to-view).
        bool tab_down = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
        if (tab_down && !tab_pressed) tab_view = !tab_view;
        tab_pressed = tab_down;

        // F5 toggles developer mode; F6 teleports through boss rooms while it is enabled.
        bool f5_down = glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS;
        if (f5_down && !f5_pressed) dev_mode = !dev_mode;
        f5_pressed = f5_down;

        bool f6_down = glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS;
        if (f6_down && !f6_pressed && dev_mode) dev_teleport_to_boss_room();
        f6_pressed = f6_down;

        bool f7_down = glfwGetKey(window, GLFW_KEY_F7) == GLFW_PRESS;
        if (f7_down && !f7_pressed && dev_mode) dev_teleport_near_exit();
        f7_pressed = f7_down;

        double move_speed = 0.04182 * 60.0 * delta; // 15% slower again (was 0.0492)

        // Facing-relative WASD in both 3D and Tab (matches look direction / green arrow)
        double new_x = player_pos_x;
        double new_y = player_pos_y;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            new_x += dir_x * move_speed;
            new_y += dir_y * move_speed;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            new_x -= dir_x * move_speed;
            new_y -= dir_y * move_speed;
        }
        // cross((dir_x,0,dir_y), (0,1,0)) = (-dir_y, 0, dir_x) — camera right
        double strafe_x = -dir_y;
        double strafe_y = dir_x;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            new_x += strafe_x * move_speed;
            new_y += strafe_y * move_speed;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            new_x -= strafe_x * move_speed;
            new_y -= strafe_y * move_speed;
        }

        new_x = clip_position(new_x, player_pos_x, true);
        new_y = clip_position(new_y, player_pos_y, false);

        if (try_move(new_x, player_pos_y)) player_pos_x = new_x;
        if (try_move(player_pos_x, new_y)) player_pos_y = new_y;
        update_player_level(); // walking the staircase hands the player between the two levels

        // Spacebar hops. Height is relative to the floor under you, so it behaves the same on
        // the stairs and the lower level. Holding the key hops again each time you land.
        bool grounded = jump_height <= 0.0 && jump_velocity <= 0.0;
        if (grounded && glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) jump_velocity = JUMP_SPEED;
        if (!grounded || jump_velocity > 0.0) {
            jump_velocity -= GRAVITY * delta;
            jump_height += jump_velocity * delta;
            if (jump_height <= 0.0) { jump_height = 0.0; jump_velocity = 0.0; }
        }

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        if (first_mouse) {
            // First frame: adopt the current cursor position so there's no startup swing.
            last_mouse_x = xpos;
            last_mouse_y = ypos;
            first_mouse = false;
        }
        double dx = xpos - last_mouse_x;
        double dy = ypos - last_mouse_y;
        last_mouse_x = xpos;
        last_mouse_y = ypos;

        double sensitivity = 0.1;
        // Same turn direction in 3D and the (now correctly-oriented) Tab map.
        yaw += dx * sensitivity;
        // Vertical aim: mouse up looks/aims up. Clamp to avoid flipping over.
        // Disabled while the Tab map (top-down) is shown.
        if (!tab_view) {
            pitch -= dy * sensitivity;
            pitch = std::clamp(pitch, -85.0, 85.0);
        }
        dir_x = std::cos(glm::radians(yaw));
        dir_y = std::sin(glm::radians(yaw));

        // One shot per click: fire on the press edge only, so holding the button does nothing
        // until you release and click again. Rate of fire is now whatever your finger manages.
        recoil = std::max(0.0, recoil - delta);
        bool fire_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (fire_down && !fire_pressed) {
            shoot();
            recoil = recoil_time;
        }
        fire_pressed = fire_down;
    }

    void render_3d() {
        glViewport(0, 0, screen_width, screen_height);
        glm::mat4 projection = glm::perspective(glm::radians(60.0f), static_cast<float>(screen_width) / screen_height, 0.01f, 100.0f);  // no wall see-through when close
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glm::vec3 camera_pos(static_cast<float>(player_pos_x), static_cast<float>(player_eye_y()), static_cast<float>(player_pos_y));
        float cp = std::cos(glm::radians(static_cast<float>(pitch)));
        float sp = std::sin(glm::radians(static_cast<float>(pitch)));
        glm::vec3 camera_front(static_cast<float>(dir_x) * cp, sp, static_cast<float>(dir_y) * cp);
        glm::vec3 camera_up(0.0f, 1.0f, 0.0f);
        glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));

        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));

        // Maze floor (with the staircase shaft cut out of it) and ceiling.
        glBindVertexArray(floor_vao);
        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
        glUniform4f(glGetUniformLocation(shader_program, "color"), 0.05f, 0.025f, 0.012f, 1.0f);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(floor_index_count), GL_UNSIGNED_INT, nullptr);

        glBindVertexArray(ceiling_vao);
        glUniform4f(glGetUniformLocation(shader_program, "color"), 0.3f, 0.3f, 0.3f, 1.0f);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        // Lower floor: its own rock, at full brightness like the walls down there. Falls back
        // to a flat colour if the image is missing.
        if (lower_floor_tex) {
            glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 1);
            glUniform4f(glGetUniformLocation(shader_program, "color"), 1.0f, 1.0f, 1.0f, 1.0f);
            glBindTexture(GL_TEXTURE_2D, lower_floor_tex);
        }
        else {
            glUniform4f(glGetUniformLocation(shader_program, "color"), 0.09f, 0.09f, 0.11f, 1.0f);
        }
        glBindVertexArray(lower_floor_vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        // The shader multiplies the texture by `color`, so it doubles as a brightness knob.
        // The maze has always been drawn dimmed - set it explicitly rather than inheriting
        // whatever the last floor/ceiling draw happened to leave behind.
        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 1);
        glUniform4f(glGetUniformLocation(shader_program, "color"), 0.3f, 0.3f, 0.3f, 1.0f);
        glBindTexture(GL_TEXTURE_2D, wall_tex);
        glBindVertexArray(wall_vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(wall_index_count), GL_UNSIGNED_INT, nullptr);

        glBindVertexArray(stair_vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(stair_index_count), GL_UNSIGNED_INT, nullptr);

        glBindTexture(GL_TEXTURE_2D, boundary_tex);
        glBindVertexArray(boundary_vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(boundary_index_count), GL_UNSIGNED_INT, nullptr);

        // Lower level: its own rock face, on the long walls and the outer shell alike. That
        // texture is already nearly black, so it is drawn at full brightness rather than dimmed.
        glUniform4f(glGetUniformLocation(shader_program, "color"), 1.0f, 1.0f, 1.0f, 1.0f);
        glBindTexture(GL_TEXTURE_2D, lower_wall_tex);
        glBindVertexArray(lower_wall_vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(lower_wall_index_count), GL_UNSIGNED_INT, nullptr);

        glBindVertexArray(lower_boundary_vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(lower_boundary_index_count), GL_UNSIGNED_INT, nullptr);

        // Exit marker: a red square on the floor of the exit cell (model is still identity).
        {
            float ex = static_cast<float>(end.first);
            float ez = static_cast<float>(end.second);
            float y = 0.02f; // just above the floor to avoid z-fighting
            float exit_verts[] = {
                ex,        y, ez,        0.0f, 0.0f,
                ex + 1.0f, y, ez,        1.0f, 0.0f,
                ex + 1.0f, y, ez + 1.0f, 1.0f, 1.0f,
                ex,        y, ez + 1.0f, 0.0f, 1.0f
            };
            glBindVertexArray(quad_vao);
            glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(exit_verts), exit_verts, GL_DYNAMIC_DRAW);
            glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
            glUniform4f(glGetUniformLocation(shader_program, "color"), 0.9f, 0.1f, 0.1f, 1.0f);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        }

        std::vector<Sprite> sprites;
        for (const auto& m : monsters) {
            // Untinted (white) normally; flash red only right after a successful hit.
            glm::vec4 tint = (m.hit_flash > 0) ? glm::vec4(1.0f, 0.25f, 0.25f, 1.0f) : glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            float centre_y = 0.5f - monster_sink(m.type); // feet on the floor, not the image edge
            sprites.emplace_back(Sprite{ glm::vec3(static_cast<float>(m.x), centre_y, static_cast<float>(m.y)), 0.5f, m.type == 1 ? monster_tex : monster2_tex, tint, true, 0.0 });
        }
        for (const auto& h : health_packs) {
            // The billboard spans centre +/- half-size vertically. Put the visible bottom of the
            // pack (not the padded edge of the image) at floor level, plus a hair to avoid a
            // flickering seam where it meets the floor.
            float floor_y = h.level == 0 ? 0.0f : static_cast<float>(lower_floor_y);
            float centre_y = floor_y + MEDPACK_HALF_SIZE * (1.0f - 2.0f * medpack_bottom_margin) + 0.003f;
            sprites.emplace_back(Sprite{ glm::vec3(static_cast<float>(h.x), centre_y, static_cast<float>(h.y)), MEDPACK_HALF_SIZE, medpack_tex, {1,1,1,1}, true, 0.0 });
        }
        for (const auto& p : projectiles) {
            glm::vec3 pp(static_cast<float>(p.x), static_cast<float>(p.z), static_cast<float>(p.y));
            // A fixed-size billboard only shrinks as 1/distance, which still leaves a chunky
            // square well down a corridor. Shrinking the quad itself with distance makes it
            // fall off as 1/distance^2, so it reads as a spark receding rather than a block.
            float d = glm::length(pp - camera_pos);
            float sz = 0.030f / (1.0f + 0.20f * d);
            sprites.emplace_back(Sprite{ pp, sz, 0, {1.0f, 0.84f, 0.0f, 1.0f}, false, 0.0 });
        }
        for (const auto& p : monster_projectiles) {
            float psize = p.from_boss ? 0.054f : 0.1f; // regular monster shots smaller; boss smaller still
            sprites.emplace_back(Sprite{ glm::vec3(static_cast<float>(p.x), static_cast<float>(p.z), static_cast<float>(p.y)), psize, 0, {0.86f, 0.2f, 0.2f, 1.0f}, false, 0.0 });
        }

        for (auto& s : sprites) {
            s.dist = glm::dot(s.pos - camera_pos, s.pos - camera_pos);
        }
        std::sort(sprites.begin(), sprites.end(), [](const Sprite& a, const Sprite& b) {
            return a.dist > b.dist;
            });

        glDepthMask(GL_FALSE);

        GLuint sprite_vao, sprite_vbo;
        glGenVertexArrays(1, &sprite_vao);
        glBindVertexArray(sprite_vao);
        glGenBuffers(1, &sprite_vbo);
        glBindBuffer(GL_ARRAY_BUFFER, sprite_vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

        for (const auto& s : sprites) {
            glm::vec3 right = glm::normalize(glm::cross(camera_front, camera_up)) * s.size;
            glm::vec3 up = camera_up * s.size;
            float verts[] = {
                s.pos.x - right.x - up.x, s.pos.y - right.y - up.y, s.pos.z - right.z - up.z, 0.0f, 0.0f,
                s.pos.x + right.x - up.x, s.pos.y + right.y - up.y, s.pos.z + right.z - up.z, 1.0f, 0.0f,
                s.pos.x + right.x + up.x, s.pos.y + right.y + up.y, s.pos.z + right.z + up.z, 1.0f, 1.0f,
                s.pos.x - right.x + up.x, s.pos.y - right.y + up.y, s.pos.z - right.z + up.z, 0.0f, 1.0f
            };
            glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
            if (s.use_tex) {
                glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 1);
                // Apply the sprite's tint (white = untinted) so leftover colors don't bleed in.
                glUniform4f(glGetUniformLocation(shader_program, "color"), s.color.x, s.color.y, s.color.z, s.color.w);
                glBindTexture(GL_TEXTURE_2D, s.tex);
            }
            else {
                glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
                glUniform4f(glGetUniformLocation(shader_program, "color"), s.color.x, s.color.y, s.color.z, s.color.w);
            }
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        }

        glDeleteBuffers(1, &sprite_vbo);
        glDeleteVertexArrays(1, &sprite_vao);
        glDepthMask(GL_TRUE);
    }

    void render_minimap() {
        // Keep the minimap square fully inside the window on any resolution.
        const int margin = 16;
        int mini_size = std::min(screen_width, screen_height) / 5;
        mini_size = std::max(64, mini_size);
        if (mini_size + 2 * margin > screen_width)
            mini_size = std::max(48, screen_width - 2 * margin);
        if (mini_size + 2 * margin > screen_height)
            mini_size = std::max(48, screen_height - 2 * margin);
        const int vp_x = margin;
        const int vp_y = screen_height - mini_size - margin; // top-left in window coords

        glViewport(vp_x, vp_y, mini_size, mini_size);
        glEnable(GL_SCISSOR_TEST);
        glScissor(vp_x, vp_y, mini_size, mini_size);

        // Soft panel background so content is clipped cleanly to the square
        glDisable(GL_DEPTH_TEST);
        glm::mat4 ortho_bg = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(ortho_bg));
        glm::mat4 id = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(id));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(id));
        draw_quad(0.0f, 0.0f, 1.0f, 1.0f, {0.15f, 0.15f, 0.18f, 0.85f});

        // Local radar centered on the player. Zoom is half-extent in world units.
        float zoom = 9.0f;
        glm::mat4 projection = glm::ortho(-zoom, zoom, -zoom, zoom, -10.0f, 10.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Player fixed at center; facing always +Y (up). Scale X by -1 so camera-right
        // (strafe D) maps to +X (right) on the minimap — matches 3D screen left/right.
        // view * p = S * R * (p - player)
        float facing = static_cast<float>(std::atan2(dir_y, dir_x));
        float rot = glm::radians(90.0f) - facing;
        glm::mat4 view = glm::mat4(1.0f);
        view = glm::scale(view, glm::vec3(-1.0f, 1.0f, 1.0f));
        view = glm::rotate(view, rot, glm::vec3(0.0f, 0.0f, 1.0f));
        view = glm::translate(view, glm::vec3(-static_cast<float>(player_pos_x), -static_cast<float>(player_pos_y), 0.0f));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));

        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));

        glLineWidth(2.0f);
        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
        glUniform4f(glGetUniformLocation(shader_program, "color"), 0.9f, 0.9f, 0.9f, 1.0f);
        // Show the walls of whichever level the player is standing on.
        glBindVertexArray(player_level == 0 ? mini_wall_vao : mini_lower_vao);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(player_level == 0 ? mini_wall_vertex_count : mini_lower_vertex_count));

        // Player marker at world pos (view centers + rotates it). Small dot + facing line.
        draw_circle(static_cast<float>(player_pos_x), static_cast<float>(player_pos_y), 0.3f, {0.0f, 1.0f, 0.0f, 1.0f});
        draw_line(static_cast<float>(player_pos_x), static_cast<float>(player_pos_y),
                  static_cast<float>(player_pos_x + dir_x * 0.6f),
                  static_cast<float>(player_pos_y + dir_y * 0.6f),
                  {0.0f, 1.0f, 0.0f, 1.0f});

        glDisable(GL_SCISSOR_TEST);
        glEnable(GL_DEPTH_TEST);
        glViewport(0, 0, screen_width, screen_height);
    }

    void render_2d() {
        glViewport(0, 0, screen_width, screen_height);
        float aspect = static_cast<float>(screen_width) / screen_height;
        float maze_size = static_cast<float>(grid_size);
        float left = -1.0f, right = maze_size + 1.0f, bottom = -1.0f, top = maze_size + 1.0f;
        if (aspect > 1.0f) {
            float extra = (aspect - 1.0f) * maze_size / 2.0f;
            left -= extra;
            right += extra;
        }
        else {
            float extra = (1.0f / aspect - 1.0f) * maze_size / 2.0f;
            bottom -= extra;
            top += extra;
        }
        // Flip the vertical axis (top/bottom swapped) so the map is a faithful top-down view
        // (+Z downward, start at top-left) instead of a mirror image of the 3D world.
        glm::mat4 projection = glm::ortho(left, right, top, bottom, -1.0f, 1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glm::mat4 view = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));

        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));

        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);

        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // The map shows whichever level the player is on; both share the same footprint.
        const auto& map_grid = grid_for(player_level);
        for (int y = 0; y < grid_size; y++) {
            for (int x = 0; x < grid_size; x++) {
                if (map_grid[y][x] == 0) {
                    draw_quad(static_cast<float>(x), static_cast<float>(y), static_cast<float>(x + 1), static_cast<float>(y + 1), { 1.0f, 1.0f, 1.0f, 1.0f });
                }
                else {
                    draw_quad(static_cast<float>(x), static_cast<float>(y), static_cast<float>(x + 1), static_cast<float>(y + 1), { 0.0f, 0.0f, 0.0f, 1.0f });
                }
            }
        }

        glLineWidth(5.0f);
        glUniform4f(glGetUniformLocation(shader_program, "color"), 0.0f, 0.0f, 0.0f, 1.0f);
        glBindVertexArray(player_level == 0 ? mini_wall_vao : mini_lower_vao);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(player_level == 0 ? mini_wall_vertex_count : mini_lower_vertex_count));

        // The staircase, marked on both levels so it's easy to find your way back.
        for (const auto& c : stair_cells) {
            draw_quad(static_cast<float>(c.first), static_cast<float>(c.second),
                      static_cast<float>(c.first + 1), static_cast<float>(c.second + 1),
                      { 0.55f, 0.75f, 1.0f, 1.0f });
        }

        // Medpacks are on both levels; show only the ones on the level being viewed.
        for (const auto& h : health_packs) {
            if (h.level != player_level) continue;
            draw_quad(static_cast<float>(h.x - 0.17), static_cast<float>(h.y - 0.17), static_cast<float>(h.x + 0.17), static_cast<float>(h.y + 0.17), { 0.68f, 0.85f, 0.9f, 1.0f });
        }

        // Everything below lives on the maze level only.
        if (player_level != 0) {
            draw_circle(static_cast<float>(player_pos_x), static_cast<float>(player_pos_y), 0.4f, { 0.0f, 1.0f, 0.0f, 1.0f });
            draw_line(static_cast<float>(player_pos_x), static_cast<float>(player_pos_y),
                static_cast<float>(player_pos_x + dir_x * 0.8), static_cast<float>(player_pos_y + dir_y * 0.8),
                { 0.0f, 0.5f, 0.0f, 1.0f });
            return;
        }

        draw_quad(static_cast<float>(end.first), static_cast<float>(end.second), static_cast<float>(end.first + 1), static_cast<float>(end.second + 1), { 1.0f, 0.24f, 0.24f, 1.0f });

        for (const auto& m : monsters) {
            glm::vec4 color = m.type == 1 ? glm::vec4(0.14f, 0.22f, 0.77f, 1.0f) : glm::vec4(0.71f, 0.12f, 0.71f, 1.0f);
            draw_circle(static_cast<float>(m.x), static_cast<float>(m.y), 0.198f, color); // 40% smaller
            float hp_frac = std::clamp(static_cast<float>(m.hp) / (m.type == 2 ? 440.0f : 300.0f), 0.0f, 1.0f);
            draw_quad(static_cast<float>(m.x - 0.33), static_cast<float>(m.y - 0.4), static_cast<float>(m.x + 0.33), static_cast<float>(m.y - 0.32), { 1.0f, 0.0f, 0.0f, 1.0f });
            draw_quad(static_cast<float>(m.x - 0.33), static_cast<float>(m.y - 0.4), static_cast<float>(m.x - 0.33 + 0.66 * hp_frac), static_cast<float>(m.y - 0.32), { 0.0f, 1.0f, 0.0f, 1.0f });
        }

        for (const auto& p : projectiles) {
            draw_circle(static_cast<float>(p.x), static_cast<float>(p.y), 0.0325f, { 1.0f, 0.84f, 0.0f, 1.0f });
        }
        for (const auto& p : monster_projectiles) {
            draw_circle(static_cast<float>(p.x), static_cast<float>(p.y), p.from_boss ? 0.0216f : 0.04f, { 0.86f, 0.2f, 0.2f, 1.0f });
        }

        // Player marker: green circle with a line pointing the way we're facing/moving.
        draw_circle(static_cast<float>(player_pos_x), static_cast<float>(player_pos_y), 0.4f, { 0.0f, 1.0f, 0.0f, 1.0f });
        draw_line(static_cast<float>(player_pos_x), static_cast<float>(player_pos_y),
            static_cast<float>(player_pos_x + dir_x * 0.8), static_cast<float>(player_pos_y + dir_y * 0.8),
            { 0.0f, 0.5f, 0.0f, 1.0f });
    }

    void render_hud(int boss_count) {
        glViewport(0, 0, screen_width, screen_height);
        glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screen_width), static_cast<float>(screen_height), 0.0f, -1.0f, 1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glm::mat4 view = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));

        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));

        glDisable(GL_DEPTH_TEST);

        float bar_w = 200.0f;
        float bar_h = 20.0f;
        float bar_x = 10.0f;
        float bar_y = static_cast<float>(screen_height) - 40.0f;
        draw_quad(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, { 1.0f, 0.0f, 0.0f, 1.0f });
        float green_w = bar_w * (player_hp / static_cast<float>(max_hp));
        draw_quad(bar_x, bar_y, bar_x + green_w, bar_y + bar_h, { 0.0f, 1.0f, 0.0f, 1.0f });

        // Draw text with white color
        draw_text("HP: " + std::to_string(player_hp) + "/" + std::to_string(max_hp), bar_x + bar_w + 10, bar_y + 5, { 1.0f, 1.0f, 1.0f, 1.0f });
        draw_text("Bosses: " + std::to_string(boss_count), bar_x, bar_y - 30, { 1.0f, 1.0f, 1.0f, 1.0f });
        if (player_level != 0)
            draw_text("Lower Level", bar_x, bar_y - 60, { 0.55f, 0.75f, 1.0f, 1.0f });

        if (dev_mode) draw_text("DEV MODE - F6: boss room  F7: near exit", 10.0f, 30.0f, { 0.75f, 0.75f, 0.75f, 1.0f });

        glEnable(GL_DEPTH_TEST);
    }

    void draw_quad(float x1, float y1, float x2, float y2, glm::vec4 color) {
        float verts[] = {
            x1, y1, 0.0f, 0.0f, 0.0f,
            x2, y1, 0.0f, 1.0f, 0.0f,
            x2, y2, 0.0f, 1.0f, 1.0f,
            x1, y2, 0.0f, 0.0f, 1.0f
        };
        glBindVertexArray(quad_vao);
        glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
        glUniform4f(glGetUniformLocation(shader_program, "color"), color.x, color.y, color.z, color.w);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }

    void draw_line(float x1, float y1, float x2, float y2, glm::vec4 color) {
        float verts[] = {
            x1, y1, 0.0f,
            x2, y2, 0.0f
        };
        glBindVertexArray(line_vao);
        glBindBuffer(GL_ARRAY_BUFFER, line_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
        glUniform4f(glGetUniformLocation(shader_program, "color"), color.x, color.y, color.z, color.w);
        glDrawArrays(GL_LINES, 0, 2);
    }

    void draw_circle(float cx, float cy, float r, glm::vec4 color) {
        const int segments = 20;
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(cx, cy, 0.0f));
        model = glm::scale(model, glm::vec3(r, r, 1.0f));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glBindVertexArray(circle_vao);
        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
        glUniform4f(glGetUniformLocation(shader_program, "color"), color.x, color.y, color.z, color.w);
        glDrawArrays(GL_TRIANGLE_FAN, 0, segments + 2);
        model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    }

    void draw_text(const std::string& text, float x, float y, glm::vec4 color) {
        float scale = 1.0f;
        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 1);
        glBindTexture(GL_TEXTURE_2D, font_tex);

        // Use white texture with color modulation
        glUniform4f(glGetUniformLocation(shader_program, "color"), color.x, color.y, color.z, color.w);

        float start_x = x;
        for (char c : text) {
            if (c < 32 || c > 127) continue;
            stbtt_bakedchar& b = cdata[c - 32];
            float verts[24] = {
                x + b.xoff * scale, y + b.yoff * scale, b.x0 / static_cast<float>(font_bitmap_w), b.y0 / static_cast<float>(font_bitmap_h),
                x + (b.xoff + b.x1 - b.x0) * scale, y + b.yoff * scale, b.x1 / static_cast<float>(font_bitmap_w), b.y0 / static_cast<float>(font_bitmap_h),
                x + (b.xoff + b.x1 - b.x0) * scale, y + (b.yoff + b.y1 - b.y0) * scale, b.x1 / static_cast<float>(font_bitmap_w), b.y1 / static_cast<float>(font_bitmap_h),
                x + b.xoff * scale, y + (b.yoff + b.y1 - b.y0) * scale, b.x0 / static_cast<float>(font_bitmap_w), b.y1 / static_cast<float>(font_bitmap_h),
                x + b.xoff * scale, y + b.yoff * scale, b.x0 / static_cast<float>(font_bitmap_w), b.y0 / static_cast<float>(font_bitmap_h),
                x + (b.xoff + b.x1 - b.x0) * scale, y + (b.yoff + b.y1 - b.y0) * scale, b.x1 / static_cast<float>(font_bitmap_w), b.y1 / static_cast<float>(font_bitmap_h)
            };
            glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            x += b.xadvance * scale;
        }

        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);

        // Check for OpenGL errors
        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error in draw_text: " << err << std::endl;
        }
    }
};

int main() {
    try {
        MazeGame game;
    }
    catch (const std::exception& e) {
        std::cerr << "Error launching game: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}