#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <stack>
#include <queue>
#include <random>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <utility>
#include <fstream>
#include <string>
#include <optional>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>

#include "net.h" // multiplayer; kept free of winsock so it can't clash with miniaudio's <windows.h>
#include "protocol.h"
#include "world.h" // the maze itself, shared with the dedicated server

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

// Command-line switches. The multiplayer ones exist mainly so two or three copies can be
// launched side by side and wired together without clicking through the menus.
struct LaunchOptions {
    bool windowed = false;   // --windowed[=left|right]: a normal window instead of fullscreen
    int window_side = 0;     // 0 = let Windows place it, 1 = left half, 2 = right half
    bool host = false;       // --host: start hosting as soon as the game opens
    std::string join;        // --join=ADDRESS: connect to a host as soon as the game opens
    int autostart = 0;       // --autostart=N: the host starts the round once N players are in
    double quit_after = 0.0; // --quit-after=SECONDS: close by itself (automated tests)
    bool test = false;       // --test: small unfocused window, mouse left free, sound off
    int test_w = 640, test_h = 360; // --test-size=WxH: that window's size (to check layouts at real sizes)
    bool test_fire = false;  // --test-fire: shoot every 2 s without input (tests the shot relay)
    bool test_spawn_near = false; // --test-spawn-near[=YAW]: Player 2 starts face to face with Player 1
    double test_spawn_yaw = 180.0; //   (or turned to YAW degrees, to look at the model from other sides)
    bool open_menu = false;  // --open-menu: start with the Esc menu showing
    bool open_sound = false; // --open-menu=sound: ...on its Sound page
    std::string screenshot;  // --screenshot=FILE: save the last frame as a .bmp before quitting
    std::string test_chat;   // --test-chat=TEXT: once playing, paste TEXT into the chat line and send it
};

struct Sprite {
    glm::vec3 pos;
    float size;
    GLuint tex;
    glm::vec4 color;
    bool use_tex;
    double dist;
};

// The game: the maze and its monsters (World, shared with the dedicated server) plus you, the
// window, the pictures, the sound and the network.
class MazeGame : private World {
private:
    double player_pos_x, player_pos_y;
    double dir_x, dir_y;
    std::vector<Projectile> projectiles;
    std::vector<Projectile> monster_projectiles;
    int max_hp = 20;            // boss shots deal 4-5, so it takes 4-5 of them to kill
    int player_hp = max_hp;
    int damage_cooldown = 0;
    double min_dist = 0.06; // get right up to a wall, but a hair in front so you can't see past it
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

    // --- Multiplayer -----------------------------------------------------------------------
    // See protocol.h for how the game is shared out. Here the authority is either this copy
    // (Mode::Host, a player hosting from the menu) or the machine it joined (Mode::Client),
    // which may be another player's copy or a dedicated server.
    static constexpr int    PVP_DAMAGE          = 5;          // 4 hits down a full-health player
    static constexpr double BEAST_SPEED         = 0.85;       // times a player's speed
    static constexpr double BEAST_FIRE_INTERVAL = 0.6;        // seconds between the Beast's shots
    static constexpr int    BEAST_MAX_HP        = 440;        // top of the boss health range
    static constexpr double RESPAWN_SECONDS     = 5.0;        // spectating after a death in multiplayer
    // Other players are a 3D soldier 0.6 tall (see build_player_model). The hit cylinder hugs
    // that figure; it was 0.22 wide for the old flat sprite, which spread its arms and gun out.
    static constexpr double PLAYER_HIT_RADIUS   = 0.16;
    static constexpr double PLAYER_HIT_HEIGHT   = 0.6;

    // Crouching (hold Ctrl): the view drops by CROUCH_DROP over CROUCH_SECONDS, easing in and
    // out, and comes back up just as fast. Crouched, you move at CROUCH_SPEED and present a
    // smaller target; the soldier model bends its knees to match.
    static constexpr double CROUCH_SECONDS      = 0.15;
    static constexpr double CROUCH_DROP         = 0.18;       // eye height 0.5 -> 0.32
    static constexpr double CROUCH_SPEED        = 0.55;       // times walking speed

    enum class Mode { Single, Host, Client };
    enum class Screen { None, Main, Host, Join, Sound, Controls };
    enum class MenuAction { None, Single, Join, Host, Sound, Exit, StartGame, CancelHost, Connect, BackFromJoin, BackFromSound,
                            TestSound, RequestStart, Controls, BackFromControls };

    struct RemotePlayer {
        bool present = false;    // in this game (lobby or round)
        bool has_state = false;  // at least one position received this round
        double x = 0, y = 0;     // latest reported position
        double rx = 0, ry = 0;   // smoothed position, used for drawing and hit tests
        double jump = 0, yaw = 0, pitch = 0;
        int level = 0, hp = 20;
        bool alive = true;
        double flash_until = 0;  // brief red tint after being hurt
        bool logged = false;     // first-position log line already written this round
        double walk_phase = 0;   // where the legs are in their stride
        double walk_amount = 0;  // 0 standing still .. 1 full stride, eased so legs settle
        bool crouching = false;  // what they last reported
        double crouch = 0;       // 0 standing .. 1 crouched, animated toward `crouching` here
    };

    // Where a vote for a new maze stands, as the host or server last described it (MSG_VOTE).
    struct VoteView {
        bool shown = false;
        uint8_t id = 0;
        int starter = 0;
        uint8_t voters = 0, yes = 0, no = 0;
        uint8_t state = VOTE_OPEN;
        double ends_at = 0.0;    // local clock: when it expires, for the countdown
        double hide_at = 0.0;    // once decided, the result shows until then
    };

    LaunchOptions opts;
    net::Session net;
    Mode mode = Mode::Single;
    Screen screen = Screen::None;
    int my_slot = 1;             // singleplayer is always slot 1
    uint32_t round_seed = 0;
    uint8_t round_id = 0;
    bool in_round = false;       // a multiplayer round is running (false in singleplayer and lobby)
    bool round_over = false;
    int round_winner = 0;
    double round_over_until = 0.0;
    uint8_t lobby_mask = 1;      // bit per occupied slot
    RemotePlayer remote[4];      // indexed by slot; the entry for my own slot is unused
    int slot_peer[4] = { -1, -1, -1, -1 }; // host: which network peer holds each slot
    int beast_boss_id = -1;      // boss Player 3 controls, or -1
    bool spectating = false;     // the Beast with no boss left to control
    // Spectating (dead and waiting to respawn, or the Beast with no boss left) is a free
    // camera: it flies through walls and floors, and fly_y is its height.
    bool free_fly = false;
    double fly_y = 0.5;
    bool crouching = false;      // Ctrl held (and allowed to crouch)
    double crouch = 0.0;         // 0 standing .. 1 crouched, moving toward `crouching`
    MapVote vote;                // host: the vote for a new maze, if one is running (rules in protocol.h)
    VoteView vote_view;          // everyone: what to show of it
    int last_damage_by = 0;      // who hurt us last, so a death names the killer
    int killed_by = 0;
    bool exit_reported = false;
    bool logged_snapshot = false;
    bool logged_shot[4] = { false, false, false, false };
    double state_timer = 0.0, monster_timer = 0.0, test_fire_timer = 0.0;
    double last_beast_shot = -10.0;
    std::vector<std::pair<std::string, double>> feed; // kill feed: text, time it disappears
    std::vector<std::string> host_ips;
    glm::mat4 last_proj = glm::mat4(1.0f), last_view = glm::mat4(1.0f); // for name labels

    // Other players' 3D figure: a soldier built in code (see build_player_model). Each part is
    // its own mesh so it can turn about a joint: the legs stride and bend to crouch, and the
    // head and arms follow where the player is aiming. Faces carry baked light as vertex
    // colours; only the camouflage trousers are textured, and their indices come first in
    // each part.
    enum PlayerPart { PART_BODY, PART_HEAD, PART_ARMS,
                      PART_THIGH_L, PART_SHIN_L, PART_BOOT_L, PART_THIGH_R, PART_SHIN_R, PART_BOOT_R, PART_COUNT };
    struct ModelPart {
        GLuint vao = 0, vbo = 0, ebo = 0;
        GLsizei textured = 0, plain = 0; // index counts
        glm::vec3 pivot{ 0.0f };         // the joint it turns about, in model metres
    };
    ModelPart player_parts[PART_COUNT];
    GLuint camo_tex = 0;
    static constexpr float PLAYER_MODEL_SCALE = 0.6f / 1.84f; // modelled 1.84 m tall, drawn 0.6
    static constexpr float LEG_HIP_Y = 0.96f, LEG_KNEE_Y = 0.50f, LEG_ANKLE_Y = 0.10f; // leg joints, model metres

    // Menu UI state
    std::string menu_status;
    bool menu_status_error = false;
    std::string join_address = "127.0.0.1";
    uint16_t join_port = net::DEFAULT_PORT;
    bool join_connecting = false, join_connected = false;
    bool server_dedicated = false;   // joined a dedicated server rather than a player's game
    bool joined_in_progress = false; // joined while a maze was under way: playing from the next one
    bool mp_maze = false;            // the maze on screen is a shared one (swapped for a fresh one on leaving)
    int menu_focus = 0;
    Screen focus_screen = Screen::None;   // for spotting the first button becoming available
    bool first_was_enabled = false;
    double menu_last_mx = -1.0, menu_last_my = -1.0;
    bool esc_pressed = false, f8_pressed = false, click_prev = false;
    bool up_prev = false, down_prev = false, enter_prev = false;
    std::string typed;           // characters typed this frame (from the char callback)
    int backspaces = 0;          // backspace presses this frame, including key repeat
    int left_presses = 0, right_presses = 0; // arrow presses this frame, for the volume slider
    // More key presses counted by the key callback and used up each frame, so nothing is
    // missed between frames and holding a key repeats where that makes sense.
    int enter_presses = 0, y_presses = 0, copy_presses = 0, paste_presses = 0;
    int f1_presses = 0, f2_presses = 0; // vote Yes / No
    int up_presses = 0, down_presses = 0, page_up_presses = 0, page_down_presses = 0;
    int home_presses = 0, end_presses = 0;
    double wheel = 0.0;          // mouse wheel movement this frame (+ = away from you)
    std::string note;            // a brief confirmation such as "Copied", shown until note_until
    double note_until = 0.0;

    // Sound option: one master volume for every sound, kept between runs.
    float master_volume = 1.0f;  // slider position, 0..1
    bool volume_dirty = false;   // changed since it was last saved
    bool slider_dragging = false;

    // --- Chat --------------------------------------------------------------------------------
    // Enter opens a line to type in; Enter again sends it to everyone (in singleplayer it just
    // shows). The newest 8 messages show along the bottom centre for 15 s each. Y opens a
    // scrollable log of everything said since the game started. It lives in memory only, so
    // it's gone when the game closes.
    static constexpr double CHAT_SHOW_SECONDS = 15.0;
    static constexpr int    CHAT_ON_SCREEN    = 8;
    struct ChatMessage {
        int from;            // sender's slot (1-3), or 0 for you in singleplayer
        std::string sender;  // "Player 2", "The Beast", "You"
        std::string text;
        std::string stamp;   // local time it arrived, "14:05"
        double shown_until;  // when it leaves the bottom of the screen
    };
    std::vector<ChatMessage> chat_history;
    bool chat_open = false;       // typing a message
    std::string chat_draft;
    bool chat_log_open = false;   // the Y window
    int chat_log_selected = -1;   // message highlighted there (Up/Down), which Ctrl+C copies
    int chat_log_top = 0;         // first line showing
    bool chat_log_follow = true;  // keep the newest line in view as messages arrive
    struct LogLine { int msg; std::string text; };
    std::vector<LogLine> chat_log_lines;   // the history, wrapped to the window's width
    std::vector<int> chat_log_first_line;  // per message: its first line in chat_log_lines
    float chat_log_wrap_w = -1.0f, chat_log_wrap_scale = -1.0f;
    double test_chat_timer = 1.5; // --test-chat: seconds of play before it sends

public:
    MazeGame(const LaunchOptions& options) : opts(options) {
        // Load the bundle first: textures, the font and audio all read through it.
        // Without it every lookup falls back to loose files, so a dev checkout still runs.
        if (assets.load("assets.dat"))
            std::cout << "assets.dat: " << assets.count() << " entries" << std::endl;
        else
            std::cerr << "assets.dat not found; falling back to loose files." << std::endl;

        load_settings();
        init_glfw();
        init_glad();
        init_opengl();
        if (!opts.test) init_audio(); // --test copies run muted, so several can run at once quietly
        init_font();
        init_draw_buffers();
        build_gun_mesh();
        build_player_model();
        load_textures();
        new_maze(random_seed(), true);

        if (opts.host) begin_hosting();
        else if (!opts.join.empty()) {
            join_address = opts.join;
            open_menu(Screen::Join);
            begin_join();
        }
        else if (opts.open_menu) open_menu(opts.open_sound ? Screen::Sound : Screen::Main);
        run();
    }

    ~MazeGame() {
        net.stop(); // tell the other players we've gone, rather than leaving them to time out
        if (volume_dirty) save_settings();
        release_level_meshes();
        for (auto& p : player_parts) {
            glDeleteVertexArrays(1, &p.vao);
            glDeleteBuffers(1, &p.vbo);
            glDeleteBuffers(1, &p.ebo);
        }
        glDeleteTextures(1, &camo_tex);
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
            apply_volume();
        }
    }

    void play_sound(const char* file) {
        if (sound_ready) ma_engine_play_sound(&sound_engine, file, nullptr);
    }

    // The slider is perceptual: loudness follows roughly the square of the gain, so halfway
    // sounds about half as loud instead of barely quieter.
    void apply_volume() {
        if (sound_ready) ma_engine_set_volume(&sound_engine, master_volume * master_volume);
    }

    void set_volume(float v) {
        v = std::round(std::clamp(v, 0.0f, 1.0f) * 100.0f) / 100.0f;
        if (v == master_volume) return;
        master_volume = v;
        volume_dirty = true;
        apply_volume();
    }

    // Settings live in %APPDATA%\MazeBeasts, so they carry over to the next release's folder.
    static std::filesystem::path settings_path() {
        const wchar_t* appdata = _wgetenv(L"APPDATA");
        std::filesystem::path dir = appdata && *appdata ? std::filesystem::path(appdata) / L"MazeBeasts"
                                                        : std::filesystem::path(L".");
        return dir / L"settings.txt";
    }

    void load_settings() {
        std::ifstream in(settings_path());
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("volume=", 0) == 0)
                master_volume = std::clamp(static_cast<float>(std::atof(line.c_str() + 7)), 0.0f, 1.0f);
        }
    }

    void save_settings() {
        volume_dirty = false;
        if (opts.test) return; // automated test copies leave your settings alone
        std::error_code ec;
        std::filesystem::create_directories(settings_path().parent_path(), ec);
        std::ofstream out(settings_path());
        if (out) out << "volume=" << master_volume << "\n";
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

    // --- Level plumbing -------------------------------------------------------------------
    // Both levels share the same coordinate space, so collision, line of sight and the map
    // all just need to be told which level's grid to consult (see World).

    // World height of the surface under a position. On the staircase this ramps smoothly
    // between the two floors; everywhere else it is whichever floor the player is on.
    double floor_y_at(double x, double y) const { return floor_height(x, y, player_level); }

    double player_eye_y() const { return floor_y_at(player_pos_x, player_pos_y) + 0.5 - CROUCH_DROP * eased(crouch) + jump_height; }

    // Smooth start and stop for 0..1 transitions (crouching).
    static double eased(double t) { t = std::clamp(t, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }

    // How much lower the top of a crouched soldier is: the model drops about as far as the view.
    static double crouch_height_drop(double amount) { return 0.19 * eased(amount); }

    void stand_up_now() {
        crouching = false;
        crouch = 0.0;
    }

    // Where the camera is: your eyes, or the free camera while spectating.
    double camera_y() const { return free_fly ? fly_y : player_eye_y(); }

    // Leave your body where it fell and float up out of it.
    void start_free_fly() {
        if (free_fly) return;
        free_fly = true;
        fly_y = player_eye_y();
        jump_height = 0.0;
        jump_velocity = 0.0;
        stand_up_now(); // Ctrl flies down now
    }

    // The spectator camera: WASD moves along where you're looking (look down and press W to
    // dive), Space rises and Ctrl or C sinks. It passes through walls and floors but stays
    // inside the outer walls, between the lower floor and the maze's ceiling.
    void fly(double speed) {
        auto held = [&](int key) { return glfwGetKey(window, key) == GLFW_PRESS; };
        double cp = std::cos(glm::radians(pitch)), sp = std::sin(glm::radians(pitch));
        double mx = 0.0, my = 0.0, mz = 0.0;
        if (held(GLFW_KEY_W)) { mx += dir_x * cp; my += sp; mz += dir_y * cp; }
        if (held(GLFW_KEY_S)) { mx -= dir_x * cp; my -= sp; mz -= dir_y * cp; }
        if (held(GLFW_KEY_D)) { mx -= dir_y; mz += dir_x; }
        if (held(GLFW_KEY_A)) { mx += dir_y; mz -= dir_x; }
        if (held(GLFW_KEY_SPACE)) my += 1.0;
        if (held(GLFW_KEY_LEFT_CONTROL) || held(GLFW_KEY_RIGHT_CONTROL) || held(GLFW_KEY_C)) my -= 1.0;
        double len = std::sqrt(mx * mx + my * my + mz * mz);
        if (len > 1.0) { mx /= len; my /= len; mz /= len; } // diagonals are no faster
        const double edge = 0.05;
        player_pos_x = std::clamp(player_pos_x + mx * speed, edge, grid_size - edge);
        player_pos_y = std::clamp(player_pos_y + mz * speed, edge, grid_size - edge);
        fly_y = std::clamp(fly_y + my * speed, lower_floor_y + 0.1, 0.92);
        player_level = fly_y < 0.0 ? 1 : 0; // the map and minimap follow the camera between floors
    }

    // The staircase cells belong to both levels, so which level the player counts as being on
    // is decided by how far down the stairs they are. By the time they can step off either end
    // the answer has already settled, which is what makes the hand-off seamless.
    void update_player_level() {
        if (is_stair_cell(static_cast<int>(player_pos_x), static_cast<int>(player_pos_y)))
            player_level = stair_progress(player_pos_x) > 0.5 ? 1 : 0;
    }

    // True if nothing blocks a straight line between two points in the maze (grid + connections).
    bool has_line_of_sight(double x0, double y0, double x1, double y1) {
        return line_of_sight(0, x0, y0, x1, y1);
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
        for (const auto& m : monsters) {
            if (m.type != 2) continue;
            if (is_beast() && m.id == beast_boss_id) continue; // the Beast doesn't hear its own music
            nearest_boss = std::min(nearest_boss, std::hypot(m.x - player_pos_x, m.y - player_pos_y));
        }
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
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* vid = glfwGetVideoMode(monitor);
        if (opts.windowed || opts.test) {
            // A normal window, so several copies can sit side by side for multiplayer testing.
            int w = opts.test ? opts.test_w : vid->width / 2 - 24;
            int h = opts.test ? opts.test_h : w * 9 / 16;
            glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
            if (opts.test) {
                glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);      // don't steal focus from whatever
                glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_FALSE); // you're doing while tests run
            }
            window = glfwCreateWindow(w, h, "MazeBeasts - 3D", nullptr, nullptr);
            if (window && opts.window_side == 1) glfwSetWindowPos(window, 12, 60);
            else if (window && opts.window_side == 2) glfwSetWindowPos(window, vid->width / 2 + 12, 60);
        }
        else {
            // Fullscreen windowed (borderless)
            screen_width = vid->width;
            screen_height = vid->height;
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
            window = glfwCreateWindow(screen_width, screen_height, "MazeBeasts - 3D", nullptr, nullptr);
            if (window) glfwSetWindowPos(window, 0, 0);
        }
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(-1);
        }
        glfwGetFramebufferSize(window, &screen_width, &screen_height);
        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetCharCallback(window, char_callback);
        glfwSetKeyCallback(window, key_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetWindowFocusCallback(window, focus_callback);
        glfwSetInputMode(window, GLFW_CURSOR, opts.test ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
    }

    // Typing into the menu's address box.
    static void char_callback(GLFWwindow* w, unsigned int codepoint) {
        MazeGame* game = static_cast<MazeGame*>(glfwGetWindowUserPointer(w));
        if (codepoint >= 32 && codepoint < 127) game->typed.push_back(static_cast<char>(codepoint));
    }

    // Keys for typing and scrolling come through here rather than polling, so nothing is
    // missed between frames, and the ones that should repeat while held do.
    static void key_callback(GLFWwindow* w, int key, int, int action, int mods) {
        MazeGame* game = static_cast<MazeGame*>(glfwGetWindowUserPointer(w));
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
        bool press = action == GLFW_PRESS;
        bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
        switch (key) {
        case GLFW_KEY_BACKSPACE: game->backspaces++; break;
        case GLFW_KEY_LEFT:      game->left_presses++; break;
        case GLFW_KEY_RIGHT:     game->right_presses++; break;
        case GLFW_KEY_UP:        game->up_presses++; break;
        case GLFW_KEY_DOWN:      game->down_presses++; break;
        case GLFW_KEY_PAGE_UP:   game->page_up_presses++; break;
        case GLFW_KEY_PAGE_DOWN: game->page_down_presses++; break;
        case GLFW_KEY_HOME:      if (press) game->home_presses++; break;
        case GLFW_KEY_END:       if (press) game->end_presses++; break;
        case GLFW_KEY_ENTER:
        case GLFW_KEY_KP_ENTER:  if (press) game->enter_presses++; break;
        case GLFW_KEY_Y:         if (press) game->y_presses++; break; // works while crouching (Ctrl) too
        case GLFW_KEY_F1:        if (press) game->f1_presses++; break;
        case GLFW_KEY_F2:        if (press) game->f2_presses++; break;
        case GLFW_KEY_C:         if (press && ctrl) game->copy_presses++; break;
        case GLFW_KEY_V:         if (ctrl) game->paste_presses++; break; // holding Ctrl+V pastes again, as in a text box
        default: break;
        }
    }

    static void scroll_callback(GLFWwindow* w, double, double dy) {
        static_cast<MazeGame*>(glfwGetWindowUserPointer(w))->wheel += dy;
    }

    // Coming back to the window (alt-tab, or clicking between copies) mustn't jerk the view,
    // and the click that brought it to the front mustn't fire a shot.
    static void focus_callback(GLFWwindow* w, int focused) {
        MazeGame* game = static_cast<MazeGame*>(glfwGetWindowUserPointer(w));
        if (focused) {
            game->first_mouse = true;
            game->fire_pressed = true;
        }
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
        // Meshes without per-vertex colours read this constant instead: plain white, so the
        // shader's colour multiply leaves them exactly as before.
        glVertexAttrib4f(2, 1.0f, 1.0f, 1.0f, 1.0f);
    }

    GLuint create_shader() {
        const char* vert_src = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        layout (location = 2) in vec4 aColor; // only the player model supplies this; others get white
        out vec2 TexCoord;
        out vec4 VertColor;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            TexCoord = aTexCoord;
            VertColor = aColor;
        }
        )";

        const char* frag_src = R"(
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;
        in vec4 VertColor;
        uniform sampler2D texture1;
        uniform int use_texture;
        uniform vec4 color;
        void main() {
            if (use_texture == 1) {
                FragColor = texture(texture1, TexCoord) * color * VertColor;
            } else {
                FragColor = color * VertColor;
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
        // The atlas is one channel of glyph coverage. Read as-is it samples as (coverage, 0, 0, 1),
        // so all text came out red inside opaque black boxes. Present it as white with the
        // coverage as alpha instead: text then takes whatever colour it's drawn with.
        const GLint swizzle[] = { GL_ONE, GL_ONE, GL_ONE, GL_RED };
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
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

    // Build a maze, its lower level, monsters and medpacks from one seed. In multiplayer every
    // machine - including a Linux dedicated server - calls this with the same seed and gets an
    // identical world (see World::generate).
    void new_maze(uint32_t seed, bool reset_facing) {
        generate(seed);
        build_meshes();
        projectiles.clear();
        monster_projectiles.clear();
        showing_win = false;
        showing_die = false;
        monster_in_view = false;
        inside_boss_room = -1;
        stop_boss_sound();
        place_player_at_spawn(reset_facing);
    }

    void regenerate_maze() { new_maze(random_seed(), false); }

    void face(double yaw_degrees) {
        yaw = yaw_degrees;
        pitch = 0.0;
        dir_x = std::cos(glm::radians(yaw));
        dir_y = std::sin(glm::radians(yaw));
    }

    void place_player_at_spawn(bool reset_facing) {
        auto c = spawn_cell(my_slot);
        player_pos_x = c.first + 0.5;
        player_pos_y = c.second + 0.5;
        player_level = 0;
        jump_height = 0.0;
        jump_velocity = 0.0;
        free_fly = false;
        stand_up_now();
        player_hp = max_hp;
        damage_cooldown = 0;
        last_damage_by = 0;
        if (reset_facing) face(open_facing_yaw(c));
        if (opts.test_spawn_near && my_slot == 2 && mode != Mode::Single) {
            // Test only: one tile in front of Player 1, looking back at them.
            player_pos_x = start.first + 1.5;
            player_pos_y = start.second + 0.5;
            face(opts.test_spawn_yaw);
        }
    }

    // Reset the player to the maze start without regenerating the maze layout.
    void respawn_player() {
        player_pos_x = start.first + 0.5;
        player_pos_y = start.second + 0.5;
        player_level = 0;
        jump_height = 0.0;
        jump_velocity = 0.0;
        stand_up_now();
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

    bool try_move(double new_x, double new_y) {
        int new_cell_x = static_cast<int>(new_x);
        int new_cell_y = static_cast<int>(new_y);
        if (new_cell_x < 0 || new_cell_x >= grid_size || new_cell_y < 0 || new_cell_y >= grid_size) return false;
        if (grid_for(player_level)[new_cell_y][new_cell_x] != 0) return false;
        // Bosses only exist on the maze level, so the Beast can't take the stairs down.
        if (is_beast() && is_stair_cell(new_cell_x, new_cell_y)) return false;
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
        p.owner = my_slot;
        projectiles.push_back(p);
        if (mode != Mode::Single && in_round) send_shot(p);
    }

    // The Beast fires boss shots from the boss's body: slow, heavy, and horizontal like the
    // bosses' own, with a cooldown so a human-driven boss can't outgun the AI ones by clicking.
    void beast_shoot() {
        double now = glfwGetTime();
        if (now - last_beast_shot < BEAST_FIRE_INTERVAL) return;
        last_beast_shot = now;
        Projectile p;
        p.x = player_pos_x; p.y = player_pos_y; p.z = 0.5;
        p.dir_x = dir_x; p.dir_y = dir_y; p.dir_z = 0.0;
        p.speed = 0.0714;
        p.from_boss = true;
        p.level = 0;
        p.owner = 3;
        monster_projectiles.push_back(p);
        send_shot(p);
    }

    // Player 1 or 2 struck by a player's shot at this point, or 0. Nobody is hit by their own
    // shots. Remote players are tested at their smoothed position, where they're drawn.
    int explorer_hit(int owner, int level, double x, double y, double z) {
        for (int s = 1; s <= 2; s++) {
            if (s == owner) continue;
            double px, py, jump, crouched;
            int lvl;
            if (s == my_slot) {
                if (showing_die) continue;
                px = player_pos_x; py = player_pos_y; lvl = player_level; jump = jump_height; crouched = crouch;
            }
            else {
                const RemotePlayer& r = remote[s];
                if (!r.present || !r.has_state || !r.alive) continue;
                px = r.rx; py = r.ry; lvl = r.level; jump = r.jump; crouched = r.crouch;
            }
            if (lvl != level || std::hypot(px - x, py - y) > PLAYER_HIT_RADIUS) continue;
            double feet = floor_height(px, py, lvl) + jump;
            if (z < feet - 0.05 || z > feet + PLAYER_HIT_HEIGHT - crouch_height_drop(crouched)) continue; // crouching ducks
            return s;
        }
        return 0;
    }

    // Player 1 or 2 struck by a monster or Beast shot, or 0. Those shots stay on the maze
    // level and have no height, exactly as monster shots always behaved against the player.
    int explorer_hit_flat(double x, double y) {
        for (int s = 1; s <= 2; s++) {
            if (s == my_slot) {
                if (!is_beast() && !showing_die && player_level == 0 && std::hypot(x - player_pos_x, y - player_pos_y) < 0.4)
                    return s;
            }
            else {
                const RemotePlayer& r = remote[s];
                if (r.present && r.has_state && r.alive && r.level == 0 && std::hypot(x - r.rx, y - r.ry) < 0.4)
                    return s;
            }
        }
        return 0;
    }

    void take_damage(int amount, int by) {
        if (showing_die || is_beast()) return;
        player_hp -= amount;
        last_damage_by = by;
    }

    // Authoritative damage to a monster: singleplayer and the host only. Clients report hits
    // with MSG_MONSTER_HIT and see the result in the next monster snapshot.
    void apply_monster_damage(int id, int dmg) {
        bool was_beast = id == beast_boss_id;
        if (damage_monster(id, dmg) && was_beast && mode == Mode::Host) reassign_beast();
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
            int hit_player = 0;
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
                // In multiplayer a shot can also strike the other explorer.
                if (!hit_monster && mode != Mode::Single)
                    hit_player = explorer_hit(it->owner, it->level, new_x, new_y, new_z);
                if (hit_wall || hit_monster || hit_player) { hit = true; break; }
                it->x = new_x; it->y = new_y; it->z = new_z;
            }
            if (hit) {
                int owner = it->owner;
                it = projectiles.erase(it);
                if (hit_monster) {
                    int id = hit_monster->id;
                    int dmg;
                    if (hit_monster->type == 1)
                        dmg = headshot ? 300 : 100; // 1 headshot or 3 body shots
                    else
                        dmg = headshot ? 150 : 50;  // 3 headshots or 7-9 body shots
                    hit_monster->hit_flash = 8; // everyone sees the flash straight away
                    // Only the shooter's machine turns the hit into damage, so it counts once.
                    if (owner == my_slot) {
                        if (mode == Mode::Client) send_monster_hit(id, dmg);
                        else apply_monster_damage(id, dmg);
                    }
                }
                // Likewise only the victim's machine applies damage to a player.
                if (hit_player && hit_player == my_slot) take_damage(PVP_DAMAGE, owner);
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
            bool hit = false;
            int hit_player = 0;
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
                // Monster shots travel the maze level only; players downstairs are out of reach.
                hit_player = explorer_hit_flat(new_x, new_y);
                if (hit_wall || hit_player) { hit = true; break; }
                it->x = new_x; it->y = new_y;
            }
            if (hit) {
                bool was_boss = it->from_boss;
                int owner = it->owner;
                it = monster_projectiles.erase(it);
                // The victim's own machine applies the damage; others just see the shot vanish.
                if (hit_player && hit_player == my_slot)
                    take_damage(was_boss ? rng.range(4, 5) : 1, owner); // 4-5 boss shots kill
            } else ++it;
        }
    }

    // Singleplayer and a hosting player run the monsters (clients follow the snapshots). They
    // shoot at the nearest living explorer on the maze level: in singleplayer, simply you.
    void update_monsters(double delta) {
        std::vector<Target> targets;
        if (!is_beast() && player_level == 0 && !showing_die) targets.push_back({ player_pos_x, player_pos_y });
        if (mode != Mode::Single) {
            for (int s = 1; s <= 2; s++) {
                if (s == my_slot) continue;
                const RemotePlayer& r = remote[s];
                if (r.present && r.has_state && r.alive && r.level == 0) targets.push_back({ r.x, r.y });
            }
        }
        std::vector<Projectile> shots;
        step_monsters(delta, beast_active() ? beast_boss_id : -1, targets, shots);
        for (const auto& p : shots) {
            monster_projectiles.push_back(p);
            if (mode == Mode::Host) send_shot(p);
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

    // Black-and-grey urban camouflage, generated rather than loaded: four tones laid down as
    // overlapping blotches of smooth noise. The noise wraps at the texture's edges, so the
    // pattern tiles without seams.
    GLuint create_camo_texture() {
        const int S = 128;
        auto hash = [](int x, int y, uint32_t seed) {
            uint32_t h = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u + seed * 2246822519u;
            h = (h ^ (h >> 13)) * 1274126177u;
            return static_cast<float>((h ^ (h >> 16)) & 0xFFFFFFu) / 16777215.0f;
        };
        auto noise = [&](float x, float y, int period, uint32_t seed) {
            int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
            float fx = x - x0, fy = y - y0;
            fx = fx * fx * (3.0f - 2.0f * fx);
            fy = fy * fy * (3.0f - 2.0f * fy);
            auto at = [&](int i, int j) { return hash(((i % period) + period) % period, ((j % period) + period) % period, seed); };
            float a = at(x0, y0), b = at(x0 + 1, y0), c = at(x0, y0 + 1), d = at(x0 + 1, y0 + 1);
            return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
        };
        auto blotch = [&](int px, int py, int period, uint32_t seed) {
            float v = 0.0f, amp = 1.0f, norm = 0.0f;
            for (int octave = 0; octave < 3; octave++, period *= 2, amp *= 0.5f) {
                v += amp * noise(px * period / static_cast<float>(S), py * period / static_cast<float>(S),
                                 period, seed + static_cast<uint32_t>(octave) * 7919u);
                norm += amp;
            }
            return v / norm;
        };
        std::vector<unsigned char> px(static_cast<size_t>(S) * S * 3);
        for (int y = 0; y < S; y++) {
            for (int x = 0; x < S; x++) {
                unsigned char g = 150;                     // light grey, about a quarter
                if (blotch(x, y, 4, 11u) > 0.46f) g = 96;  // mid grey
                if (blotch(x, y, 4, 23u) > 0.61f) g = 52;  // charcoal
                if (blotch(x, y, 5, 37u) > 0.64f) g = 16;  // black
                unsigned char* p = &px[(static_cast<size_t>(y) * S + x) * 3];
                p[0] = p[1] = p[2] = g;
            }
        }
        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, S, S, 0, GL_RGB, GL_UNSIGNED_BYTE, px.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        return tex;
    }

    // The other players: a soldier in urban-camouflage trousers, a grey shirt and black boots,
    // with black skin and a helmet and chin strap in the old Player 2 blue. Built once from
    // boxes and ellipsoids, in metres with the feet at the origin, facing +Z; that makes
    // their right-hand side -X. Light is baked into vertex colours, as on your own gun.
    void build_player_model() {
        camo_tex = create_camo_texture();

        struct Vert { glm::vec3 p; glm::vec2 uv; glm::vec4 c; };
        struct Build { std::vector<Vert> verts; std::vector<unsigned int> tex, plain; };
        Build parts[PART_COUNT];

        const glm::vec3 light = glm::normalize(glm::vec3(-0.30f, 0.80f, 0.52f));
        auto lit = [&](glm::vec3 col, glm::vec3 n, bool emissive) {
            float s = emissive ? 1.0f : 0.36f + 0.64f * std::max(0.0f, glm::dot(glm::normalize(n), light));
            return glm::vec4(col * s, 1.0f);
        };

        const glm::vec3 WHITE(1.0f);                      // camouflage: the texture is the colour
        const glm::vec3 SHIRT(0.50f, 0.51f, 0.53f);
        const glm::vec3 COLLAR(0.40f, 0.41f, 0.43f);
        const glm::vec3 SKIN(0.10f, 0.09f, 0.09f);
        const glm::vec3 BOOT(0.07f, 0.07f, 0.075f);
        const glm::vec3 PAD(0.17f, 0.17f, 0.18f);
        const glm::vec3 BELT(0.11f, 0.11f, 0.11f);
        const glm::vec3 BUCKLE(0.40f, 0.40f, 0.42f);
        const glm::vec3 GUN(0.14f, 0.15f, 0.16f);
        const glm::vec3 TRIM(0.33f, 0.97f, 0.97f);       // the cyan on your own gun
        const glm::vec3 BLUE(player_color(2));            // what Player 2's sprite used to be

        // A box, optionally rotated. Camouflage faces tile the texture every 0.5 m, each face
        // starting from a different spot so the pattern doesn't repeat across them.
        int face_seed = 0;
        auto box = [&](int part, glm::vec3 c, glm::vec3 h, glm::mat3 rot, glm::vec3 col, bool camo, bool emissive = false) {
            const glm::vec3 nrm[6] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
            const int idx[6][4] = { {1,5,7,3}, {4,0,2,6}, {2,3,7,6}, {4,5,1,0}, {5,4,6,7}, {0,1,3,2} };
            glm::vec3 p[8];
            for (int i = 0; i < 8; i++)
                p[i] = c + rot * glm::vec3((i & 1) ? h.x : -h.x, (i & 2) ? h.y : -h.y, (i & 4) ? h.z : -h.z);
            Build& b = parts[part];
            for (int f = 0; f < 6; f++) {
                glm::vec4 colour = lit(col, rot * nrm[f], emissive);
                face_seed++;
                glm::vec2 off(std::fmod(face_seed * 0.37f, 1.0f), std::fmod(face_seed * 0.61f, 1.0f));
                float u = glm::length(p[idx[f][1]] - p[idx[f][0]]) / 0.5f;
                float v = glm::length(p[idx[f][3]] - p[idx[f][0]]) / 0.5f;
                const glm::vec2 uv[4] = { off, off + glm::vec2(u, 0.0f), off + glm::vec2(u, v), off + glm::vec2(0.0f, v) };
                unsigned int base = static_cast<unsigned int>(b.verts.size());
                for (int k = 0; k < 4; k++) b.verts.push_back({ p[idx[f][k]], uv[k], colour });
                auto& list = camo ? b.tex : b.plain;
                for (unsigned int i : { 0u, 1u, 2u, 0u, 2u, 3u }) list.push_back(base + i);
            }
        };
        const glm::mat3 I(1.0f);
        auto upright = [&](int part, glm::vec3 c, glm::vec3 h, glm::vec3 col, bool camo = false) { box(part, c, h, I, col, camo); };

        // A limb: a box running from joint a to joint b.
        auto limb = [&](int part, glm::vec3 a, glm::vec3 b, float hw, float hd, glm::vec3 col, bool camo = false) {
            glm::vec3 axis = b - a;
            float len = glm::length(axis);
            glm::vec3 y = axis / len;
            glm::vec3 ref = std::fabs(y.x) < 0.9f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 0.0f, 1.0f);
            glm::vec3 x = glm::normalize(ref - y * glm::dot(ref, y));
            box(part, (a + b) * 0.5f, glm::vec3(hw, len * 0.5f, hd), glm::mat3(x, y, glm::cross(x, y)), col, camo);
        };

        // A smooth-shaded slice of an ellipsoid between two latitudes (degrees, 90 = the top).
        auto ellipsoid = [&](int part, glm::vec3 c, glm::vec3 r, float lat0, float lat1, glm::vec3 col) {
            const int lon_segs = 18, lat_segs = 7;
            Build& b = parts[part];
            unsigned int base = static_cast<unsigned int>(b.verts.size());
            for (int i = 0; i <= lat_segs; i++) {
                float lat = glm::radians(lat0 + (lat1 - lat0) * i / lat_segs);
                for (int j = 0; j <= lon_segs; j++) {
                    float lon = glm::two_pi<float>() * j / lon_segs;
                    glm::vec3 d(std::cos(lat) * std::sin(lon), std::sin(lat), std::cos(lat) * std::cos(lon));
                    b.verts.push_back({ c + d * r, glm::vec2(0.0f), lit(col, d / r, false) });
                }
            }
            for (int i = 0; i < lat_segs; i++) {
                for (int j = 0; j < lon_segs; j++) {
                    unsigned int a0 = base + i * (lon_segs + 1) + j, a1 = a0 + 1;
                    unsigned int b0 = a0 + lon_segs + 1, b1 = b0 + 1;
                    for (unsigned int k : { a0, a1, b1, a0, b1, b0 }) b.plain.push_back(k);
                }
            }
        };

        // ---- Legs: camouflage trousers, knee pads and boots. Each leg is three parts so it can
        // bend: the thigh swings from the hip, the shin bends at the knee, and the boot flexes
        // at the ankle to stay flat on the floor.
        for (int side = 0; side < 2; side++) {
            float x = side == 0 ? 0.105f : -0.105f;
            int thigh = side == 0 ? PART_THIGH_L : PART_THIGH_R;
            int shin = side == 0 ? PART_SHIN_L : PART_SHIN_R;
            int boot = side == 0 ? PART_BOOT_L : PART_BOOT_R;
            limb(thigh, { x, 0.50f, 0.0f }, { x, 0.96f, 0.0f }, 0.090f, 0.102f, WHITE, true);
            limb(shin, { x, 0.11f, 0.0f }, { x, 0.52f, 0.0f }, 0.074f, 0.084f, WHITE, true);
            upright(shin, { x, 0.50f, 0.088f }, { 0.060f, 0.060f, 0.016f }, PAD);
            upright(boot, { x, 0.060f, 0.035f }, { 0.068f, 0.060f, 0.135f }, BOOT);
            player_parts[thigh].pivot = { x, LEG_HIP_Y, 0.0f };
            player_parts[shin].pivot = { x, LEG_KNEE_Y, 0.0f };
            player_parts[boot].pivot = { x, LEG_ANKLE_Y, 0.0f };
        }

        // ---- Body: hips, belt, grey shirt, shoulders and neck.
        upright(PART_BODY, { 0.0f, 0.975f, 0.0f }, { 0.195f, 0.090f, 0.112f }, WHITE, true);
        upright(PART_BODY, { 0.0f, 1.070f, 0.0f }, { 0.200f, 0.030f, 0.118f }, BELT);
        upright(PART_BODY, { 0.0f, 1.070f, 0.119f }, { 0.032f, 0.022f, 0.006f }, BUCKLE);
        upright(PART_BODY, { 0.0f, 1.190f, 0.0f }, { 0.190f, 0.095f, 0.110f }, SHIRT);
        upright(PART_BODY, { 0.0f, 1.370f, 0.0f }, { 0.215f, 0.115f, 0.125f }, SHIRT);
        ellipsoid(PART_BODY, { 0.212f, 1.425f, 0.0f }, { 0.066f, 0.060f, 0.076f }, -90.0f, 90.0f, SHIRT);
        ellipsoid(PART_BODY, { -0.212f, 1.425f, 0.0f }, { 0.066f, 0.060f, 0.076f }, -90.0f, 90.0f, SHIRT);
        upright(PART_BODY, { 0.0f, 1.495f, 0.0f }, { 0.085f, 0.015f, 0.075f }, COLLAR);
        upright(PART_BODY, { 0.0f, 1.540f, 0.0f }, { 0.050f, 0.045f, 0.050f }, SKIN);

        // ---- Head: black, under a blue helmet with a flared rim, and a blue chin strap.
        const glm::vec3 head_c(0.0f, 1.640f, 0.005f), head_r(0.085f, 0.105f, 0.100f);
        ellipsoid(PART_HEAD, head_c, head_r, -90.0f, 90.0f, SKIN);
        const glm::vec3 helmet_c(0.0f, 1.665f, -0.005f), helmet_r(0.125f, 0.160f, 0.140f);
        ellipsoid(PART_HEAD, helmet_c, helmet_r, 0.0f, 90.0f, BLUE);
        {
            // The rim: a short skirt below the dome, flaring out a little as it goes down.
            const int segs = 18;
            const float drop = 0.030f, flare = 1.08f;
            Build& b = parts[PART_HEAD];
            unsigned int base = static_cast<unsigned int>(b.verts.size());
            for (int j = 0; j <= segs; j++) {
                float lon = glm::two_pi<float>() * j / segs;
                glm::vec3 d(std::sin(lon), 0.0f, std::cos(lon));
                glm::vec3 top = helmet_c + d * helmet_r;
                glm::vec3 bottom = helmet_c + glm::vec3(d.x * helmet_r.x * flare, -drop, d.z * helmet_r.z * flare);
                glm::vec3 n = glm::normalize(glm::vec3(d.x / helmet_r.x, 0.0f, d.z / helmet_r.z)) + glm::vec3(0.0f, 0.3f, 0.0f);
                glm::vec4 colour = lit(BLUE, n, false);
                b.verts.push_back({ top, glm::vec2(0.0f), colour });
                b.verts.push_back({ bottom, glm::vec2(0.0f), colour });
            }
            for (int j = 0; j < segs; j++) {
                unsigned int a0 = base + 2 * j, a1 = a0 + 1, b0 = a0 + 2, b1 = a0 + 3;
                for (unsigned int k : { a0, a1, b1, a0, b1, b0 }) b.plain.push_back(k);
            }
        }
        {
            // The chin strap runs from under the rim on one side, beneath the chin, to the
            // other side, lying just off the surface of the head.
            const glm::vec3 side(1.0f, 0.0f, 0.0f), under = glm::normalize(glm::vec3(0.0f, -0.85f, 0.53f));
            const glm::vec3 across = glm::normalize(glm::cross(side, under));
            const int segs = 16;
            const float half_w = 0.011f;
            Build& b = parts[PART_HEAD];
            unsigned int base = static_cast<unsigned int>(b.verts.size());
            for (int i = 0; i <= segs; i++) {
                float t = glm::pi<float>() * i / segs;
                glm::vec3 d = glm::normalize(std::cos(t) * side + std::sin(t) * under);
                float to_surface = 1.0f / std::sqrt(glm::dot(d / head_r, d / head_r));
                glm::vec3 p = head_c + d * to_surface * 1.06f;
                glm::vec4 colour = lit(BLUE, d / (head_r * head_r), false);
                b.verts.push_back({ p + across * half_w, glm::vec2(0.0f), colour });
                b.verts.push_back({ p - across * half_w, glm::vec2(0.0f), colour });
            }
            for (int i = 0; i < segs; i++) {
                unsigned int a0 = base + 2 * i, a1 = a0 + 1, b0 = a0 + 2, b1 = a0 + 3;
                for (unsigned int k : { a0, a1, b1, a0, b1, b0 }) b.plain.push_back(k);
            }
        }

        // ---- Arms and rifle, held at the ready on the right; they pitch together to aim.
        const glm::vec3 r_shoulder(-0.225f, 1.43f, 0.0f), r_elbow(-0.26f, 1.16f, -0.02f), r_hand(-0.105f, 1.21f, 0.19f);
        const glm::vec3 l_shoulder(0.225f, 1.43f, 0.0f), l_elbow(0.12f, 1.19f, 0.22f), l_hand(-0.085f, 1.25f, 0.38f);
        limb(PART_ARMS, r_shoulder, r_elbow, 0.058f, 0.062f, SHIRT);
        limb(PART_ARMS, r_elbow, r_hand, 0.045f, 0.048f, SKIN);
        limb(PART_ARMS, l_shoulder, l_elbow, 0.058f, 0.062f, SHIRT);
        limb(PART_ARMS, l_elbow, l_hand, 0.045f, 0.048f, SKIN);
        upright(PART_ARMS, r_hand, { 0.032f, 0.036f, 0.032f }, SKIN);
        upright(PART_ARMS, l_hand, { 0.032f, 0.036f, 0.032f }, SKIN);
        const float gx = -0.105f;
        upright(PART_ARMS, { gx, 1.275f, 0.250f }, { 0.030f, 0.045f, 0.160f }, GUN);  // receiver
        upright(PART_ARMS, { gx, 1.290f, 0.520f }, { 0.016f, 0.016f, 0.120f }, GUN);  // barrel
        upright(PART_ARMS, { gx, 1.260f, 0.040f }, { 0.026f, 0.042f, 0.070f }, GUN);  // stock
        upright(PART_ARMS, { gx, 1.195f, 0.300f }, { 0.022f, 0.060f, 0.030f }, GUN);  // magazine
        upright(PART_ARMS, { gx, 1.330f, 0.200f }, { 0.012f, 0.012f, 0.040f }, GUN);  // sight
        box(PART_ARMS, { gx + 0.031f, 1.290f, 0.280f }, { 0.002f, 0.008f, 0.120f }, I, TRIM, false, true); // glowing
        box(PART_ARMS, { gx - 0.031f, 1.290f, 0.280f }, { 0.002f, 0.008f, 0.120f }, I, TRIM, false, true);

        player_parts[PART_HEAD].pivot = { 0.0f, 1.52f, 0.0f };
        player_parts[PART_ARMS].pivot = { 0.0f, 1.42f, 0.0f };

        for (int i = 0; i < PART_COUNT; i++) {
            const Build& b = parts[i];
            std::vector<float> data;
            data.reserve(b.verts.size() * 9);
            for (const Vert& v : b.verts)
                data.insert(data.end(), { v.p.x, v.p.y, v.p.z, v.uv.x, v.uv.y, v.c.r, v.c.g, v.c.b, v.c.a });
            std::vector<unsigned int> inds = b.tex;
            inds.insert(inds.end(), b.plain.begin(), b.plain.end());

            ModelPart& m = player_parts[i];
            m.textured = static_cast<GLsizei>(b.tex.size());
            m.plain = static_cast<GLsizei>(b.plain.size());
            glGenVertexArrays(1, &m.vao);
            glBindVertexArray(m.vao);
            glGenBuffers(1, &m.vbo);
            glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
            glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
            glGenBuffers(1, &m.ebo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, inds.size() * sizeof(unsigned int), inds.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), nullptr);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(5 * sizeof(float)));
        }
        glBindVertexArray(0);
    }

    // One soldier standing at (x, feet height, y), facing along yaw, with the head and rifle
    // following the pitch, the legs mid-stride, and `crouch` (0..1, already eased) bending the
    // knees. The tint is multiplied over everything.
    void draw_player_model(double x, double feet, double y, double yaw_deg, double pitch_deg,
                           double walk_phase, double walk_amount, double crouch_amount, glm::vec4 tint) {
        // The model faces +Z; turn that onto the heading (cos yaw, sin yaw) in the maze.
        float yaw_r = glm::radians(static_cast<float>(yaw_deg));
        glm::mat4 base = glm::translate(glm::mat4(1.0f), glm::vec3(static_cast<float>(x), static_cast<float>(feet), static_cast<float>(y)));
        base = glm::rotate(base, std::atan2(std::cos(yaw_r), std::sin(yaw_r)), glm::vec3(0.0f, 1.0f, 0.0f));
        base = glm::scale(base, glm::vec3(PLAYER_MODEL_SCALE));

        // Turning about +X by a negative angle tips the front (+Z) upward - or, for a leg,
        // swings the knee forward.
        auto turn = [](const glm::mat4& m, glm::vec3 pivot, float angle) {
            if (angle == 0.0f) return m;
            return glm::translate(m, pivot) * glm::rotate(glm::mat4(1.0f), angle, glm::vec3(1.0f, 0.0f, 0.0f))
                 * glm::translate(glm::mat4(1.0f), -pivot);
        };
        float aim = glm::radians(std::clamp(static_cast<float>(pitch_deg), -60.0f, 60.0f));
        float swing = glm::radians(34.0f) * static_cast<float>(walk_amount) * std::sin(static_cast<float>(walk_phase));

        // Crouching: the thighs come forward (to 75 degrees) and the shins lean back (to 60), so
        // the hips drop; the boots stay flat. Everything above the hips drops with them and tips
        // forward a little, while the whole figure shifts back so the feet stay where they were.
        const float c = static_cast<float>(crouch_amount);
        const float thigh_a = glm::radians(75.0f) * c, shin_a = glm::radians(60.0f) * c;
        const float thigh_len = LEG_HIP_Y - LEG_KNEE_Y, shin_len = LEG_KNEE_Y - LEG_ANKLE_Y;
        const float drop = LEG_HIP_Y - (LEG_ANKLE_Y + thigh_len * std::cos(thigh_a) + shin_len * std::cos(shin_a));
        const float ankle_ahead = thigh_len * std::sin(thigh_a) - shin_len * std::sin(shin_a);
        const glm::mat4 hips = glm::translate(base, glm::vec3(0.0f, -drop, -ankle_ahead));
        const float lean = glm::radians(20.0f) * c;
        const glm::mat4 upper = turn(hips, glm::vec3(0.0f, LEG_HIP_Y, 0.0f), lean);

        glm::mat4 mats[PART_COUNT];
        mats[PART_BODY] = upper;
        mats[PART_HEAD] = turn(upper, player_parts[PART_HEAD].pivot, -aim * 0.6f - lean); // still looking ahead
        mats[PART_ARMS] = turn(upper, player_parts[PART_ARMS].pivot, -aim - lean);        // still aiming where they aim
        const int legs[2][3] = { { PART_THIGH_L, PART_SHIN_L, PART_BOOT_L }, { PART_THIGH_R, PART_SHIN_R, PART_BOOT_R } };
        for (int side = 0; side < 2; side++) {
            const int* leg = legs[side];
            float leg_swing = side == 0 ? swing : -swing;
            mats[leg[0]] = turn(hips, player_parts[leg[0]].pivot, -thigh_a + leg_swing);
            mats[leg[1]] = turn(mats[leg[0]], player_parts[leg[1]].pivot, thigh_a + shin_a); // knee: shin ends up leaning back
            mats[leg[2]] = turn(mats[leg[1]], player_parts[leg[2]].pivot, -shin_a);           // ankle: boot level again
        }

        glUniform4f(glGetUniformLocation(shader_program, "color"), tint.r, tint.g, tint.b, tint.a);
        glBindTexture(GL_TEXTURE_2D, camo_tex);
        for (int i = 0; i < PART_COUNT; i++) {
            const ModelPart& p = player_parts[i];
            const glm::mat4& m = mats[i];
            glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(m));
            glBindVertexArray(p.vao);
            if (p.textured) {
                glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 1);
                glDrawElements(GL_TRIANGLES, p.textured, GL_UNSIGNED_INT, nullptr);
            }
            if (p.plain) {
                glUniform1i(glGetUniformLocation(shader_program, "use_texture"), 0);
                glDrawElements(GL_TRIANGLES, p.plain, GL_UNSIGNED_INT, (void*)(p.textured * sizeof(unsigned int)));
            }
        }
        glm::mat4 id = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(id));
        glVertexAttrib4f(2, 1.0f, 1.0f, 1.0f, 1.0f); // back to white for meshes without colours
    }

    // GPU buffers for the current maze. Rebuilding a maze used to allocate a fresh set without
    // freeing the old one; with a new maze every multiplayer round that leak adds up.
    bool meshes_built = false;

    void release_level_meshes() {
        if (!meshes_built) return;
        GLuint vaos[] = { wall_vao, boundary_vao, floor_vao, ceiling_vao, lower_wall_vao,
                          lower_boundary_vao, lower_floor_vao, stair_vao, mini_wall_vao, mini_lower_vao };
        GLuint bufs[] = { wall_vbo, wall_ebo, boundary_vbo, boundary_ebo, floor_vbo, floor_ebo,
                          ceiling_vbo, ceiling_ebo, lower_wall_vbo, lower_wall_ebo, lower_boundary_vbo,
                          lower_boundary_ebo, lower_floor_vbo, lower_floor_ebo, stair_vbo, stair_ebo,
                          mini_wall_vbo, mini_lower_vbo };
        glDeleteVertexArrays(static_cast<GLsizei>(std::size(vaos)), vaos);
        glDeleteBuffers(static_cast<GLsizei>(std::size(bufs)), bufs);
        meshes_built = false;
    }

    void build_meshes() {
        release_level_meshes();
        meshes_built = true;
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

    // =========================================================================================
    // Multiplayer
    // =========================================================================================

    bool is_beast() const { return mode != Mode::Single && in_round && my_slot == 3; }

    // True while a person is driving the Beast's boss, so the host's AI must leave it alone.
    bool beast_active() const {
        return mode != Mode::Single && in_round && beast_boss_id >= 0 && (lobby_mask & (1 << 3)) != 0;
    }

    int player_count() const {
        int n = 0;
        for (int s = 1; s <= 3; s++) if (lobby_mask & (1 << s)) n++;
        return n;
    }

    static std::string player_name(int slot) { return slot == 3 ? "the Beast" : "Player " + std::to_string(slot); }

    static std::string capitalised(std::string s) {
        if (!s.empty() && s[0] >= 'a' && s[0] <= 'z') s[0] = static_cast<char>(s[0] - 'a' + 'A');
        return s;
    }

    static glm::vec4 player_color(int slot) {
        switch (slot) {
        case 1:  return { 0.40f, 1.00f, 0.50f, 1.0f }; // green
        case 2:  return { 0.45f, 0.75f, 1.00f, 1.0f }; // blue
        default: return { 0.85f, 0.45f, 1.00f, 1.0f }; // purple, the Beast
        }
    }

    std::string death_text(int victim, int killer) const {
        std::string who = victim == my_slot ? "You" : capitalised(player_name(victim));
        std::string verb = victim == my_slot ? " were killed by " : " was killed by ";
        if (killer == 0) return who + verb + "a monster";
        if (killer == my_slot) return who + verb + "you";
        return who + verb + player_name(killer);
    }

    void add_feed(const std::string& text) {
        feed.push_back({ text, glfwGetTime() + 5.0 });
        if (feed.size() > 4) feed.erase(feed.begin());
        log("[feed] " + text);
    }

    // Test aid: write the frame just drawn to a .bmp (bottom-up rows, which BMP expects anyway).
    void save_screenshot(const std::string& path) {
        int w = screen_width, h = screen_height;
        int row = (w * 3 + 3) & ~3;
        std::vector<unsigned char> px(static_cast<size_t>(row) * h);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, px.data());
        std::ofstream f(path, std::ios::binary);
        if (!f) return;
        uint32_t data_size = static_cast<uint32_t>(px.size());
        unsigned char hdr[54] = { 'B', 'M' };
        auto put32 = [&](int o, uint32_t v) { for (int i = 0; i < 4; i++) hdr[o + i] = static_cast<unsigned char>(v >> (8 * i)); };
        put32(2, 54 + data_size); put32(10, 54); put32(14, 40);
        put32(18, static_cast<uint32_t>(w)); put32(22, static_cast<uint32_t>(h));
        hdr[26] = 1; hdr[28] = 24; put32(34, data_size);
        f.write(reinterpret_cast<const char*>(hdr), 54);
        f.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
        log("[test] screenshot saved to " + path);
    }

    static void log(const std::string& line) { std::cout << line << std::endl; }

    // ---- Sending ----------------------------------------------------------------------------

    // Host: to every client. Client: to the host, which relays it to everyone else.
    void send_msg(const std::vector<uint8_t>& msg, bool reliable) {
        if (mode == Mode::Host) net.broadcast(msg, reliable);
        else if (mode == Mode::Client) net.send_to_host(msg, reliable);
    }

    void send_shot(const Projectile& p) {
        if (mode == Mode::Single || !in_round) return;
        net::Writer w;
        w.put<uint8_t>(MSG_SHOT).put<uint8_t>(round_id).put<uint8_t>(static_cast<uint8_t>(p.owner))
         .put<uint8_t>(static_cast<uint8_t>(p.level)).put<uint8_t>(p.from_boss ? 1 : 0)
         .put<float>(static_cast<float>(p.x)).put<float>(static_cast<float>(p.y)).put<float>(static_cast<float>(p.z))
         .put<float>(static_cast<float>(p.dir_x)).put<float>(static_cast<float>(p.dir_y))
         .put<float>(static_cast<float>(p.dir_z)).put<float>(static_cast<float>(p.speed));
        send_msg(w.buf, true);
    }

    void send_monster_hit(int id, int dmg) {
        net::Writer w;
        w.put<uint8_t>(MSG_MONSTER_HIT).put<uint8_t>(round_id).put<int32_t>(id).put<int32_t>(dmg);
        net.send_to_host(w.buf, true);
    }

    void send_state() {
        uint8_t flags = static_cast<uint8_t>(((!showing_die && !spectating) ? STATE_ALIVE : 0) | (crouching ? STATE_CROUCHING : 0));
        net::Writer w;
        w.put<uint8_t>(MSG_STATE).put<uint8_t>(round_id).put<uint8_t>(static_cast<uint8_t>(my_slot))
         .put<uint8_t>(static_cast<uint8_t>(player_level)).put<uint8_t>(flags)
         .put<float>(static_cast<float>(player_pos_x)).put<float>(static_cast<float>(player_pos_y))
         .put<float>(static_cast<float>(jump_height)).put<float>(static_cast<float>(yaw))
         .put<float>(static_cast<float>(pitch)).put<int16_t>(static_cast<int16_t>(std::clamp(player_hp, -999, 999)));
        send_msg(w.buf, false);
    }

    // Host: where every monster is, 20 times a second. Unreliable on purpose - each snapshot
    // is complete, so a lost one is simply replaced by the next.
    void send_monsters() {
        size_t n = std::min<size_t>(monsters.size(), 512);
        net::Writer w;
        w.put<uint8_t>(MSG_MONSTERS).put<uint8_t>(round_id).put<uint16_t>(static_cast<uint16_t>(n));
        for (size_t i = 0; i < n; i++) {
            const Monster& m = monsters[i];
            w.put<int32_t>(m.id).put<float>(static_cast<float>(m.x)).put<float>(static_cast<float>(m.y))
             .put<int32_t>(m.hp).put<uint8_t>(static_cast<uint8_t>(m.type))
             .put<uint8_t>(static_cast<uint8_t>(std::clamp(m.hit_flash, 0, 255)));
        }
        net.broadcast(w.buf, false);
    }

    void send_periodic(double delta) {
        state_timer -= delta;
        if (state_timer <= 0.0) { state_timer = STATE_INTERVAL; send_state(); }
        if (mode == Mode::Host) {
            monster_timer -= delta;
            if (monster_timer <= 0.0) { monster_timer = MONSTER_INTERVAL; send_monsters(); }
        }
    }

    void broadcast_lobby() {
        net::Writer w;
        w.put<uint8_t>(MSG_LOBBY).put<uint8_t>(lobby_mask);
        net.broadcast(w.buf, true);
    }

    // ---- Voting for a new maze ------------------------------------------------------------
    // Whoever runs the game (here, when hosting; otherwise the host's game or the dedicated
    // server) keeps the vote with the shared rules in protocol.h. Everyone else asks, votes,
    // and shows what they're told.

    // F8, or "votemap" in chat.
    void request_vote() {
        if (!in_round) return;
        if (mode == Mode::Host) host_vote_start(my_slot);
        else if (mode == Mode::Client) {
            net::Writer w;
            w.put<uint8_t>(MSG_VOTE_START);
            net.send_to_host(w.buf, true);
        }
    }

    // F1 (Yes) or F2 (No), if there's a vote you can still vote in.
    void cast_vote(bool yes) {
        uint8_t me = static_cast<uint8_t>(1 << my_slot);
        const VoteView& v = vote_view;
        if (!v.shown || v.state != VOTE_OPEN || !(v.voters & me) || ((v.yes | v.no) & me)) return;
        if (mode == Mode::Host) {
            host_vote_cast(my_slot, yes);
            return;
        }
        net::Writer w;
        w.put<uint8_t>(MSG_VOTE_CAST).put<uint8_t>(yes ? 1 : 0);
        net.send_to_host(w.buf, true);
        if (yes) vote_view.yes |= me; else vote_view.no |= me; // shown at once; the next MSG_VOTE confirms it
    }

    // What the vote panel shows: from MSG_VOTE, or the host's own copy of it.
    void on_vote(uint8_t id, int starter, uint8_t voters, uint8_t yes, uint8_t no, uint8_t state, double seconds_left) {
        double now = glfwGetTime();
        vote_view.shown = true;
        vote_view.id = id;
        vote_view.starter = starter;
        vote_view.voters = voters;
        vote_view.yes = yes;
        vote_view.no = no;
        vote_view.state = state;
        vote_view.ends_at = now + seconds_left;
        if (state != VOTE_OPEN) vote_view.hide_at = now + 3.0; // the result stays up briefly
    }

    // A line from the game itself, "Game: ...": to everyone, or to one player.
    void game_notice(const std::string& text) {
        add_chat(0, text);
        net::Writer w;
        w.put<uint8_t>(MSG_CHAT).put<uint8_t>(0).put_text(text);
        net.broadcast(w.buf, true);
    }

    void game_notice_to(int slot, const std::string& text) {
        if (slot == my_slot) { add_chat(0, text); return; }
        if (slot < 1 || slot > 3 || slot_peer[slot] < 0) return;
        net::Writer w;
        w.put<uint8_t>(MSG_CHAT).put<uint8_t>(0).put_text(text);
        net.send(slot_peer[slot], w.buf, true);
    }

    void host_broadcast_vote(VoteState state) {
        double left = vote.seconds_left(glfwGetTime());
        net::Writer w;
        w.put<uint8_t>(MSG_VOTE).put<uint8_t>(vote.id).put<uint8_t>(static_cast<uint8_t>(vote.starter))
         .put<uint8_t>(vote.voters).put<uint8_t>(vote.yes).put<uint8_t>(vote.no).put<uint8_t>(state)
         .put<uint16_t>(static_cast<uint16_t>(std::min(65000.0, left * 1000.0)));
        net.broadcast(w.buf, true);
        on_vote(vote.id, vote.starter, vote.voters, vote.yes, vote.no, state, left);
    }

    void host_vote_start(int slot) {
        if (!in_round || round_over) { game_notice_to(slot, "Votes can only be started during a maze."); return; }
        int r = vote.start(slot, lobby_mask, glfwGetTime());
        if (r == -1) { game_notice_to(slot, "A vote is already running: F1 = Yes, F2 = No."); return; }
        if (r > 0) {
            game_notice_to(slot, "Nobody answered your last " + std::to_string(VOTE_SOLO_LIMIT)
                                 + " votes. You can start another in " + minutes_seconds(r) + ".");
            return;
        }
        game_notice(capitalised(player_name(slot)) + " started a vote for a new maze: F1 = Yes, F2 = No");
        host_broadcast_vote(VOTE_OPEN);
        host_settle_vote();
    }

    void host_vote_cast(int slot, bool yes) {
        if (!in_round || !vote.cast(slot, yes)) return;
        host_broadcast_vote(VOTE_OPEN);
        host_settle_vote();
    }

    // Every frame while hosting, and after each change: pass, fail or expire once decided.
    void host_settle_vote() {
        if (!vote.active) return;
        VoteState state = vote.check(glfwGetTime());
        if (state == VOTE_OPEN) return;
        host_broadcast_vote(state);
        if (state == VOTE_PASSED) {
            game_notice("Vote passed: here's a new maze!");
            host_start_round();
        }
        else game_notice(state == VOTE_FAILED ? "Vote failed: no new maze." : "Vote expired: not enough players voted.");
    }

    // ---- Receiving --------------------------------------------------------------------------

    void pump_network() {
        if (!net.active()) return;
        for (const auto& e : net.poll()) {
            if (!net.active()) break; // a handler below may have ended the session
            switch (e.type) {
            case net::Event::Type::Connected:    on_peer_connected(); break;
            case net::Event::Type::Disconnected: on_peer_disconnected(e.peer); break;
            case net::Event::Type::Received:
                if (mode == Mode::Host) host_receive(e.peer, e.data);
                else client_receive(e.data);
                break;
            }
        }
    }

    int slot_of_peer(int peer) const {
        for (int s = 2; s <= 3; s++) if (slot_peer[s] == peer) return s;
        return -1;
    }

    void on_peer_connected() {
        // The host waits for this HELLO before it hands out a slot.
        if (mode != Mode::Client) return;
        net::Writer w;
        w.put<uint8_t>(MSG_HELLO).put<uint8_t>(PROTOCOL_VERSION);
        net.send_to_host(w.buf, true);
        menu_status = "Connected. Joining...";
        menu_status_error = false;
    }

    void on_peer_disconnected(int peer) {
        if (mode == Mode::Client) {
            if (!join_connected) {
                // No Connected ever arrived: nobody answered at that address.
                leave_multiplayer("Could not reach " + join_target() + " (UDP).");
            }
            else {
                // A server that stopped, or a connection that dropped (ENet notices within ~10 s).
                leave_multiplayer(server_dedicated ? "Lost the connection to the server." : "The host closed the game.");
                open_menu(Screen::Main);
            }
            return;
        }
        if (mode != Mode::Host) return;
        int slot = slot_of_peer(peer);
        if (slot < 0) return; // never finished joining
        slot_peer[slot] = -1;
        remote[slot] = RemotePlayer{};
        lobby_mask = static_cast<uint8_t>(lobby_mask & ~(1 << slot));
        vote.remove_player(slot); // a running vote is settled without them
        log("[net] player " + std::to_string(slot) + " left");
        if (in_round) {
            add_feed(capitalised(player_name(slot)) + " left the game");
            if (slot == 3 && beast_boss_id >= 0) {
                // Nobody drives that boss any more: hand it back to its AI.
                beast_boss_id = -1;
                net::Writer w;
                w.put<uint8_t>(MSG_BEAST_BOSS).put<uint8_t>(round_id).put<int32_t>(-1);
                net.broadcast(w.buf, true);
            }
        }
        else if (slot == 2 && slot_peer[3] >= 0) {
            // Keep a Player 2 so the host can still start: whoever was Player 3 moves up.
            int p = slot_peer[3];
            slot_peer[3] = -1;
            slot_peer[2] = p;
            remote[2] = remote[3];
            remote[3] = RemotePlayer{};
            lobby_mask = static_cast<uint8_t>((lobby_mask & ~(1 << 3)) | (1 << 2));
            net::Writer w;
            w.put<uint8_t>(MSG_ASSIGN).put<uint8_t>(2);
            net.send(p, w.buf, true);
        }
        broadcast_lobby();
    }

    // Reads a STATE message into remote[slot]. expected_slot is the sender's slot on the host
    // (so nobody can move someone else) and -1 on a client, which trusts the host's relay.
    bool read_state(net::Reader& r, int expected_slot) {
        int slot = r.get<uint8_t>();
        int level = r.get<uint8_t>();
        uint8_t flags = r.get<uint8_t>();
        float x = r.get<float>(), y = r.get<float>(), jump = r.get<float>();
        float yw = r.get<float>(), pt = r.get<float>();
        int hp = r.get<int16_t>();
        if (!r.ok || slot < 1 || slot > 3 || slot == my_slot) return false;
        if (expected_slot >= 0 && slot != expected_slot) return false;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(jump) || !std::isfinite(yw) || !std::isfinite(pt)) return false;
        x = std::clamp(x, 0.0f, static_cast<float>(grid_size) - 0.001f);
        y = std::clamp(y, 0.0f, static_cast<float>(grid_size) - 0.001f);
        RemotePlayer& p = remote[slot];
        bool teleport = !p.has_state || (level != 0) != (p.level != 0) || std::hypot(x - p.x, y - p.y) > 2.0;
        bool alive = (flags & STATE_ALIVE) != 0;
        if (p.has_state && alive && hp < p.hp) p.flash_until = glfwGetTime() + 0.15;
        p.x = x; p.y = y;
        if (teleport) { p.rx = x; p.ry = y; }
        p.crouching = (flags & STATE_CROUCHING) != 0; // animated toward in smooth_remote_players
        if (teleport) p.crouch = p.crouching ? 1.0 : 0.0;
        p.jump = std::clamp(jump, 0.0f, 1.0f);
        p.yaw = yw; p.pitch = pt;
        p.level = level != 0 ? 1 : 0;
        p.hp = hp;
        p.alive = alive;
        p.present = true;
        p.has_state = true;
        if (!p.logged) { p.logged = true; log("[mp] receiving " + player_name(slot) + "'s position"); }
        // The host drives the Beast's boss from Player 3's reported position.
        if (mode == Mode::Host && slot == 3 && beast_boss_id >= 0) {
            if (Monster* m = find_monster(beast_boss_id)) {
                m->net_x = x; m->net_y = y;
                if (teleport) { m->x = x; m->y = y; }
            }
        }
        return true;
    }

    bool read_shot(net::Reader& r, int expected_owner) {
        Projectile p;
        int owner = r.get<uint8_t>();
        p.level = r.get<uint8_t>() ? 1 : 0;
        p.from_boss = r.get<uint8_t>() != 0;
        float v[7];
        for (float& f : v) f = r.get<float>();
        if (!r.ok || owner > 3 || owner == my_slot) return false;
        if (expected_owner >= 0 && owner != expected_owner) return false;
        for (float f : v) if (!std::isfinite(f)) return false;
        p.x = std::clamp(v[0], 0.0f, static_cast<float>(grid_size)); p.y = std::clamp(v[1], 0.0f, static_cast<float>(grid_size));
        p.z = v[2];
        p.dir_x = v[3]; p.dir_y = v[4]; p.dir_z = v[5];
        p.speed = std::clamp(v[6], 0.01f, 1.0f);
        p.owner = owner;
        // Monster and Beast shots behave like monster shots; the explorers' like player shots.
        if (owner == 0 || owner == 3) monster_projectiles.push_back(p);
        else projectiles.push_back(p);
        if (!logged_shot[owner]) {
            logged_shot[owner] = true;
            log(std::string("[mp] shot received from ") + (owner == 0 ? "a monster" : player_name(owner)));
        }
        return true;
    }

    // Replace the monster list with the host's snapshot. Known monsters keep their drawn
    // position and glide toward the new one; ones missing from the snapshot are dead.
    void read_snapshot(net::Reader& r) {
        int n = r.get<uint16_t>();
        if (!r.ok || n > 512) return;
        std::vector<Monster> next;
        next.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; i++) {
            int id = r.get<int32_t>();
            float x = r.get<float>(), y = r.get<float>();
            int hp = r.get<int32_t>();
            int type = r.get<uint8_t>();
            int flash = r.get<uint8_t>();
            if (!r.ok || !std::isfinite(x) || !std::isfinite(y)) return;
            Monster m{};
            if (const Monster* old = find_monster(id)) m = *old;
            else { m.x = x; m.y = y; m.target_x = x; m.target_y = y; }
            m.id = id;
            m.net_x = std::clamp(x, 0.0f, static_cast<float>(grid_size));
            m.net_y = std::clamp(y, 0.0f, static_cast<float>(grid_size));
            m.hp = hp;
            m.type = type == 2 ? 2 : 1;
            m.hit_flash = std::max(m.hit_flash, flash);
            next.push_back(m);
        }
        monsters.swap(next);
        if (!logged_snapshot) {
            logged_snapshot = true;
            log("[mp] first monster snapshot: " + std::to_string(monsters.size()) + " monsters, "
                + std::to_string(boss_count()) + " bosses");
        }
    }

    void remove_pack(int id) {
        health_packs.erase(std::remove_if(health_packs.begin(), health_packs.end(),
                           [id](const HealthPack& h) { return h.id == id; }), health_packs.end());
    }

    void host_receive(int peer, const std::vector<uint8_t>& data) {
        net::Reader r(data);
        uint8_t type = r.get<uint8_t>();
        int slot = slot_of_peer(peer);

        if (type == MSG_HELLO) {
            uint8_t version = r.get<uint8_t>();
            if (slot >= 0) return; // already joined
            uint8_t reason = 0;
            int free_slot = slot_peer[2] < 0 ? 2 : (slot_peer[3] < 0 ? 3 : -1);
            if (!r.ok || version != PROTOCOL_VERSION) reason = REJECT_VERSION;
            else if (in_round) reason = REJECT_IN_PROGRESS;
            else if (free_slot < 0) reason = REJECT_FULL;
            if (reason) {
                net::Writer w;
                w.put<uint8_t>(MSG_REJECT).put<uint8_t>(reason);
                net.send(peer, w.buf, true);
                net.drop(peer);
                log("[net] turned away a player (reason " + std::to_string(reason) + ")");
                return;
            }
            slot_peer[free_slot] = peer;
            lobby_mask = static_cast<uint8_t>(lobby_mask | (1 << free_slot));
            remote[free_slot] = RemotePlayer{};
            remote[free_slot].present = true;
            net::Writer w;
            w.put<uint8_t>(MSG_WELCOME).put<uint8_t>(static_cast<uint8_t>(free_slot)).put<uint32_t>(round_seed)
             .put<uint8_t>(0); // flags: a player's game, not a dedicated server; never mid-round
            net.send(peer, w.buf, true);
            broadcast_lobby();
            log("[net] player " + std::to_string(free_slot) + " joined (" + std::to_string(player_count()) + "/3)");
            return;
        }

        if (slot < 0) return; // hasn't said hello yet
        if (type == MSG_CHAT) {
            // Shown here, and passed on to everyone else labelled with who said it.
            std::string text = trim_spaces(clean_chat_text(r.get_text()));
            if (!r.ok || text.empty()) return;
            add_chat(slot, text);
            net::Writer w;
            w.put<uint8_t>(MSG_CHAT).put<uint8_t>(static_cast<uint8_t>(slot)).put_text(text);
            net.broadcast(w.buf, true, peer);
            return;
        }
        if (type == MSG_VOTE_START) { host_vote_start(slot); return; }
        if (type == MSG_VOTE_CAST) {
            uint8_t choice = r.get<uint8_t>();
            if (r.ok) host_vote_cast(slot, choice != 0);
            return;
        }
        uint8_t rnd = r.get<uint8_t>();
        if (!r.ok || !in_round || rnd != round_id) return; // from a previous maze, or no round yet

        switch (type) {
        case MSG_STATE:
            if (read_state(r, slot)) net.broadcast(data, false, peer);
            break;
        case MSG_SHOT:
            if (read_shot(r, slot)) net.broadcast(data, true, peer);
            break;
        case MSG_MONSTER_HIT: {
            int id = r.get<int32_t>();
            int dmg = r.get<int32_t>();
            if (r.ok && dmg > 0 && dmg <= 300) apply_monster_damage(id, dmg);
            break;
        }
        case MSG_DEATH: {
            int victim = r.get<uint8_t>();
            int killer = r.get<uint8_t>();
            if (!r.ok || victim != slot || killer > 3) break;
            net.broadcast(data, true, peer);
            add_feed(death_text(victim, killer));
            remote[victim].alive = false;
            break;
        }
        case MSG_PICKUP: {
            int id = r.get<int32_t>();
            if (!r.ok) break;
            bool exists = std::any_of(health_packs.begin(), health_packs.end(), [id](const HealthPack& h) { return h.id == id; });
            if (!exists) break; // someone got there first
            remove_pack(id);
            net::Writer w;
            w.put<uint8_t>(MSG_PACK_GONE).put<uint8_t>(round_id).put<int32_t>(id);
            net.broadcast(w.buf, true);
            break;
        }
        case MSG_REACHED_EXIT: {
            int who = r.get<uint8_t>();
            // The host has the final say on whether every boss is really dead.
            if (r.ok && who == slot && !round_over && boss_count() == 0) end_round(slot);
            break;
        }
        default:
            break;
        }
    }

    void client_receive(const std::vector<uint8_t>& data) {
        net::Reader r(data);
        uint8_t type = r.get<uint8_t>();
        switch (type) {
        case MSG_WELCOME: {
            int slot = r.get<uint8_t>();
            uint32_t seed = r.get<uint32_t>();
            uint8_t flags = r.get<uint8_t>();
            bool dedicated = (flags & WELCOME_DEDICATED) != 0;
            // A player's game is always Player 1 itself; on a dedicated server anyone can be.
            if (!r.ok || slot < (dedicated ? 1 : 2) || slot > 3) return;
            my_slot = slot;
            round_seed = seed;
            server_dedicated = dedicated;
            joined_in_progress = (flags & WELCOME_IN_PROGRESS) != 0;
            join_connecting = false;
            join_connected = true;
            menu_status.clear();
            lobby_mask = static_cast<uint8_t>((dedicated ? 0 : 1 << 1) | (1 << slot)); // the LOBBY message follows
            glfwSetWindowTitle(window, ("MazeBeasts - Player " + std::to_string(slot)).c_str());
            log("[net] joined " + std::string(dedicated ? "a dedicated server" : "a player's game") + " as player "
                + std::to_string(slot) + (joined_in_progress ? ", playing from the next maze" : ""));
            return;
        }
        case MSG_REJECT: {
            int reason = r.get<uint8_t>();
            std::string why = reason == REJECT_IN_PROGRESS ? "That game has already started."
                            : reason == REJECT_FULL ? "That game is full (3 players)."
                            : "That host or server is running a different version of MazeBeasts.";
            leave_multiplayer(why);
            return;
        }
        case MSG_ASSIGN: {
            int slot = r.get<uint8_t>();
            if (!r.ok || slot < (server_dedicated ? 1 : 2) || slot > 3) return;
            my_slot = slot;
            glfwSetWindowTitle(window, ("MazeBeasts - Player " + std::to_string(slot)).c_str());
            log("[net] now player " + std::to_string(slot));
            return;
        }
        case MSG_CHAT: {
            int from = r.get<uint8_t>(); // 0: the host or server's own notices
            std::string text = trim_spaces(clean_chat_text(r.get_text()));
            if (r.ok && from <= 3 && !text.empty()) add_chat(from, text);
            return;
        }
        case MSG_VOTE: {
            uint8_t id = r.get<uint8_t>();
            int starter = r.get<uint8_t>();
            uint8_t voters = r.get<uint8_t>(), yes = r.get<uint8_t>(), no = r.get<uint8_t>(), state = r.get<uint8_t>();
            int ms = r.get<uint16_t>();
            if (r.ok && in_round && state <= VOTE_EXPIRED) on_vote(id, starter, voters, yes, no, state, ms / 1000.0);
            return;
        }
        case MSG_TO_LOBBY: {
            // Dedicated server: every explorer left this maze, so it ended without a winner.
            if (!in_round) return;
            in_round = false;
            round_over = false;
            spectating = false;
            showing_die = false;
            free_fly = false;
            beast_boss_id = -1;
            feed.clear();
            stop_boss_sound();
            open_menu(Screen::Join);
            menu_status = "The explorers left, so that maze ended.";
            menu_status_error = false;
            log("[mp] back to the lobby");
            return;
        }
        case MSG_LOBBY: {
            uint8_t mask = r.get<uint8_t>();
            if (!r.ok) return;
            if (in_round) {
                for (int s = 1; s <= 3; s++) {
                    if (s != my_slot && (lobby_mask & (1 << s)) && !(mask & (1 << s))) {
                        add_feed(capitalised(player_name(s)) + " left the game");
                        remote[s] = RemotePlayer{};
                    }
                }
            }
            lobby_mask = mask;
            return;
        }
        case MSG_START: {
            uint8_t rnd = r.get<uint8_t>();
            uint32_t seed = r.get<uint32_t>();
            int beast = r.get<int32_t>();
            uint8_t mask = r.get<uint8_t>();
            if (!r.ok || !join_connected) return;
            round_id = rnd;
            round_seed = seed;
            lobby_mask = mask;
            new_maze(seed, true);
            beast_boss_id = beast;
            begin_round_local();
            return;
        }
        default:
            break;
        }

        // Everything else belongs to the current round.
        uint8_t rnd = r.get<uint8_t>();
        if (!r.ok || !in_round || rnd != round_id) return;
        switch (type) {
        case MSG_STATE:    read_state(r, -1); break;
        case MSG_SHOT:     read_shot(r, -1); break;
        case MSG_MONSTERS: read_snapshot(r); break;
        case MSG_DEATH: {
            int victim = r.get<uint8_t>();
            int killer = r.get<uint8_t>();
            if (!r.ok || victim < 1 || victim > 3 || victim == my_slot) break;
            add_feed(death_text(victim, killer));
            remote[victim].alive = false;
            break;
        }
        case MSG_PACK_GONE: {
            int id = r.get<int32_t>();
            if (r.ok) remove_pack(id);
            break;
        }
        case MSG_ROUND_OVER: {
            int winner = r.get<uint8_t>();
            if (!r.ok) break;
            round_over = true;
            round_winner = winner;
            round_over_until = glfwGetTime() + ROUND_OVER_SECONDS;
            log("[mp] round over, won by " + player_name(winner));
            break;
        }
        case MSG_BEAST_BOSS: {
            int id = r.get<int32_t>();
            if (!r.ok) break;
            beast_boss_id = id;
            if (my_slot == 3) take_beast_control();
            break;
        }
        default:
            break;
        }
    }

    // ---- Sessions and rounds ------------------------------------------------------------------

    void begin_hosting() {
        std::string err;
        if (!net.start_host(net::DEFAULT_PORT, 2, err)) {
            menu_status = err;
            menu_status_error = true;
            open_menu(Screen::Main);
            log("[net] " + err);
            return;
        }
        mode = Mode::Host;
        my_slot = 1;
        lobby_mask = 1 << 1;
        in_round = false;
        round_id = 0;
        for (auto& p : remote) p = RemotePlayer{};
        for (int& p : slot_peer) p = -1;
        vote = MapVote{};
        vote_view = VoteView{};
        round_seed = random_seed(); // shown in the lobby; the first round uses it
        host_ips = net.local_ipv4_addresses();
        menu_status.clear();
        menu_status_error = false;
        open_menu(Screen::Host);
        glfwSetWindowTitle(window, "MazeBeasts - Player 1 (host)");
        log("[net] hosting on UDP port " + std::to_string(net::DEFAULT_PORT) + ", maze seed " + std::to_string(round_seed));
    }

    // "1.2.3.4", "1.2.3.4:29180" or a host name, optionally with a port. On failure, says why.
    static bool split_address(const std::string& text, std::string& host, uint16_t& port, std::string& error) {
        host = text;
        port = net::DEFAULT_PORT;
        size_t colon = text.rfind(':');
        if (colon == std::string::npos) return true;
        host = text.substr(0, colon);
        std::string digits = text.substr(colon + 1);
        bool numeric = !digits.empty() && digits.size() <= 5
                    && std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; });
        long value = numeric ? std::atol(digits.c_str()) : 0;
        if (host.empty() || host.find(':') != std::string::npos || value < 1 || value > 65535) {
            error = "\"" + text + "\" isn't an address. Use an IP like 1.2.3.4 or 1.2.3.4:29180.";
            return false;
        }
        port = static_cast<uint16_t>(value);
        return true;
    }

    std::string join_target() const { return join_address.find(':') == std::string::npos ? join_address + ":" + std::to_string(join_port) : join_address; }

    void begin_join() {
        std::string err, host;
        uint16_t port = net::DEFAULT_PORT;
        if (!split_address(join_address, host, port, err) || !net.start_client(host, port, err)) {
            menu_status = err;
            menu_status_error = true;
            log("[net] " + err);
            return;
        }
        join_port = port;
        mode = Mode::Client;
        in_round = false;
        vote_view = VoteView{};
        join_connecting = true;
        join_connected = false;
        lobby_mask = 1 << 1;
        for (auto& p : remote) p = RemotePlayer{};
        menu_status = "Connecting to " + join_target() + "...";
        menu_status_error = false;
        log("[net] connecting to " + join_target());
    }

    // Back to singleplayer: close the connection, and if a multiplayer maze was in play, give
    // the player a fresh solo one - a shared maze isn't a singleplayer game.
    void leave_multiplayer(const std::string& why) {
        bool had_round = mp_maze;
        net.stop();
        mode = Mode::Single;
        my_slot = 1;
        in_round = false;
        round_over = false;
        spectating = false;
        beast_boss_id = -1;
        lobby_mask = 1 << 1;
        join_connecting = false;
        join_connected = false;
        server_dedicated = false;
        joined_in_progress = false;
        for (auto& p : remote) p = RemotePlayer{};
        for (int& p : slot_peer) p = -1;
        feed.clear();
        vote = MapVote{};
        vote_view = VoteView{};
        mp_maze = false;
        if (had_round) new_maze(random_seed(), true);
        glfwSetWindowTitle(window, "MazeBeasts - 3D");
        menu_status = why;
        menu_status_error = !why.empty();
        if (!why.empty()) log("[net] " + why);
    }

    void host_start_round() {
        if (mode != Mode::Host) return;
        vote.cancel(); // a new maze makes any vote moot
        if (round_id > 0) round_seed = random_seed(); // later rounds get a fresh maze
        round_id++;
        new_maze(round_seed, true);
        beast_boss_id = -1;
        if (slot_peer[3] >= 0) {
            std::vector<int> bosses;
            for (const auto& m : monsters) if (m.type == 2) bosses.push_back(m.id);
            if (!bosses.empty())
                beast_boss_id = bosses[static_cast<size_t>(rng.range(0, static_cast<int>(bosses.size()) - 1))];
        }
        net::Writer w;
        w.put<uint8_t>(MSG_START).put<uint8_t>(round_id).put<uint32_t>(round_seed)
         .put<int32_t>(beast_boss_id).put<uint8_t>(lobby_mask);
        net.broadcast(w.buf, true);
        begin_round_local();
    }

    void begin_round_local() {
        in_round = true;
        mp_maze = true;
        joined_in_progress = false;
        menu_status.clear();
        if (vote_view.state == VOTE_OPEN) vote_view.shown = false; // a passed vote's result stays up a moment
        round_over = false;
        round_winner = 0;
        exit_reported = false;
        spectating = false;
        logged_snapshot = false;
        for (bool& b : logged_shot) b = false;
        killed_by = 0;
        last_damage_by = 0;
        for (int s = 1; s <= 3; s++) {
            remote[s] = RemotePlayer{};
            remote[s].present = s != my_slot && (lobby_mask & (1 << s)) != 0;
        }
        feed.clear();
        state_timer = 0.0;
        monster_timer = 0.0;
        if (my_slot == 3) take_beast_control();
        if (screen != Screen::None) close_menu();
        log("[mp] round " + std::to_string(round_id) + " started as player " + std::to_string(my_slot)
            + ": seed " + std::to_string(round_seed) + ", world checksum " + std::to_string(world_checksum())
            + ", " + std::to_string(boss_count()) + " bosses, Beast boss id " + std::to_string(beast_boss_id));
    }

    // Player 3 becomes the boss they've been given: the camera moves into it.
    void take_beast_control() {
        const Monster* m = find_monster(beast_boss_id);
        if (!m) {
            spectating = true;
            start_free_fly(); // fly around and watch until the next maze
            log("[mp] no boss left to control - spectating");
            return;
        }
        spectating = false;
        free_fly = false;
        player_pos_x = m->x;
        player_pos_y = m->y;
        player_level = 0;
        jump_height = 0.0;
        jump_velocity = 0.0;
        face(open_facing_yaw({ static_cast<int>(m->x), static_cast<int>(m->y) }));
        log("[mp] controlling boss " + std::to_string(beast_boss_id));
    }

    // Host: the Beast's boss died. Give Player 3 another living boss, if there is one.
    void reassign_beast() {
        beast_boss_id = -1;
        if (slot_peer[3] >= 0) {
            std::vector<int> bosses;
            for (const auto& m : monsters) if (m.type == 2) bosses.push_back(m.id);
            if (!bosses.empty()) {
                beast_boss_id = bosses[static_cast<size_t>(rng.range(0, static_cast<int>(bosses.size()) - 1))];
                if (Monster* m = find_monster(beast_boss_id)) { m->net_x = m->x; m->net_y = m->y; }
            }
        }
        net::Writer w;
        w.put<uint8_t>(MSG_BEAST_BOSS).put<uint8_t>(round_id).put<int32_t>(beast_boss_id);
        net.broadcast(w.buf, true);
        log("[mp] the Beast's boss died; now controlling boss " + std::to_string(beast_boss_id));
    }

    void end_round(int winner) {
        round_over = true;
        round_winner = winner;
        round_over_until = glfwGetTime() + ROUND_OVER_SECONDS;
        net::Writer w;
        w.put<uint8_t>(MSG_ROUND_OVER).put<uint8_t>(round_id).put<uint8_t>(static_cast<uint8_t>(winner));
        net.broadcast(w.buf, true);
        log("[mp] round over, won by " + player_name(winner));
    }

    void smooth_remote_players(double delta) {
        double k = std::min(1.0, delta * 15.0);
        for (auto& r : remote) {
            if (!r.has_state) continue;
            double ox = r.rx, oy = r.ry;
            r.rx += (r.x - r.rx) * k;
            r.ry += (r.y - r.ry) * k;
            // Legs stride in step with the ground covered: one full cycle every 0.62 units,
            // easing to a stop rather than freezing mid-step.
            double moved = std::hypot(r.rx - ox, r.ry - oy);
            double stride = delta > 0.0 ? std::clamp(moved / delta / 2.0, 0.0, 1.0) : 0.0;
            if (r.jump > 0.02) stride = 0.0; // legs together in the air
            r.walk_amount += (stride - r.walk_amount) * std::min(1.0, delta * 8.0);
            r.walk_phase = std::fmod(r.walk_phase + moved * glm::two_pi<double>() / 0.62, glm::two_pi<double>());
            // Crouching and standing take the same time as they do for the player doing it.
            double step = delta / CROUCH_SECONDS;
            r.crouch = r.crouching ? std::min(1.0, r.crouch + step) : std::max(0.0, r.crouch - step);
        }
    }

    // Clients don't run monster AI; they glide monsters toward the host's latest snapshot.
    void smooth_monsters(double delta) {
        double k = std::min(1.0, delta * 12.0);
        for (auto& m : monsters) {
            m.x += (m.net_x - m.x) * k;
            m.y += (m.net_y - m.y) * k;
            if (m.hit_flash > 0) m.hit_flash--;
        }
    }

    // =========================================================================================
    // Chat
    // =========================================================================================

    std::string clipboard_text() {
        const char* s = glfwGetClipboardString(window); // nullptr when the clipboard holds no text
        return s ? std::string(s) : std::string();
    }

    void copy_to_clipboard(const std::string& text) {
        if (text.empty()) return;
        glfwSetClipboardString(window, text.c_str());
        flash_note("Copied: " + (text.size() > 40 ? text.substr(0, 37) + "..." : text));
    }

    // A brief confirmation, shown in place of the hint line wherever you are.
    void flash_note(const std::string& text) {
        note = text;
        note_until = glfwGetTime() + 2.0;
    }

    bool note_showing() const { return !note.empty() && glfwGetTime() < note_until; }

    // Slot 0 is the game itself: notices about votes and the like.
    std::string chat_sender(int from) const {
        if (from < 1 || from > 3) return "Game";
        if (mode == Mode::Single) return "You";
        return capitalised(player_name(from));
    }

    // Everything said, by anyone, comes through here: into the history (and so the log), and
    // onto the bottom of the screen for CHAT_SHOW_SECONDS.
    void add_chat(int from, const std::string& text) {
        std::time_t t = std::time(nullptr);
        char stamp[8] = "";
        if (const std::tm* tm = std::localtime(&t)) std::strftime(stamp, sizeof(stamp), "%H:%M", tm);
        chat_history.push_back({ from, chat_sender(from), text, stamp, glfwGetTime() + CHAT_SHOW_SECONDS });
        log("[chat] " + chat_history.back().sender + ": " + text);
    }

    // Ctrl+V into the chat line: the text made safe for the font, up to the length limit.
    void chat_paste(const std::string& text) {
        if (chat_draft.size() < CHAT_MAX_CHARS)
            chat_draft += clean_chat_text(text, CHAT_MAX_CHARS - chat_draft.size());
    }

    // Enter on the chat line: send it (unless it's blank) and close the line. "votemap" isn't
    // said out loud: it asks for a vote on a new maze (in singleplayer, it just makes one).
    void submit_chat() {
        std::string text = trim_spaces(clean_chat_text(chat_draft));
        chat_open = false;
        chat_draft.clear();
        if (text.empty()) return;
        std::string command = text;
        for (char& c : command) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (command == "votemap" || command == "/votemap") {
            if (mode == Mode::Single) regenerate_maze();
            else request_vote();
            return;
        }
        add_chat(my_slot, text);
        net::Writer w;
        if (mode == Mode::Host) {
            w.put<uint8_t>(MSG_CHAT).put<uint8_t>(static_cast<uint8_t>(my_slot)).put_text(text);
            net.broadcast(w.buf, true);
        }
        else if (mode == Mode::Client && join_connected) {
            w.put<uint8_t>(MSG_CHAT).put_text(text);
            net.send_to_host(w.buf, true);
        }
    }

    void copy_selected_chat() {
        if (chat_log_selected >= 0 && chat_log_selected < static_cast<int>(chat_history.size()))
            copy_to_clipboard(chat_history[chat_log_selected].text);
    }

    // Once a frame, before the player's own controls (which the open chat line takes the
    // keyboard from). The menus handle their own keys; the log's scrolling is handled where
    // it's drawn, since that's where its size is known.
    void update_chat(double delta) {
        if (screen != Screen::None) return;
        if (!opts.test_chat.empty() && test_chat_timer > 0.0 && (mode == Mode::Single || in_round)) {
            test_chat_timer -= delta;
            if (test_chat_timer <= 0.0) {
                chat_open = true;
                chat_paste(opts.test_chat); // the path Ctrl+V takes, without touching your clipboard
                submit_chat();
            }
        }
        if (chat_open) {
            for (char c : typed) if (chat_draft.size() < CHAT_MAX_CHARS) chat_draft.push_back(c);
            for (int i = 0; i < backspaces && !chat_draft.empty(); i++) chat_draft.pop_back();
            for (int i = 0; i < paste_presses; i++) chat_paste(clipboard_text());
            if (copy_presses) {
                // Copies what you've typed; with nothing typed, the message picked in the log.
                if (!chat_draft.empty()) copy_to_clipboard(chat_draft);
                else if (chat_log_open) copy_selected_chat();
            }
            if (enter_presses) submit_chat();
        }
        else {
            // Letters typed while playing (W, A, S, D...) don't carry into a chat line opened
            // later: only what's typed after Enter counts.
            if (enter_presses) { chat_open = true; chat_draft.clear(); }
            else if (y_presses) {
                chat_log_open = !chat_log_open;
                if (chat_log_open) {
                    chat_log_selected = static_cast<int>(chat_history.size()) - 1; // the newest
                    chat_log_follow = true;
                }
            }
            if (copy_presses && chat_log_open) copy_selected_chat();
        }
    }

    // =========================================================================================
    // Menu
    // =========================================================================================

    struct Button { std::string label; MenuAction action; bool enabled; };

    void set_mouse_captured(bool captured) {
        if (opts.test) captured = false; // test copies never grab the mouse
        glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
        first_mouse = true; // no view jump from wherever the cursor wandered meanwhile
    }

    void open_menu(Screen s) {
        bool was_closed = screen == Screen::None;
        screen = s;
        menu_focus = 0;
        if (was_closed) {
            set_mouse_captured(false);
            int ww, wh;
            glfwGetWindowSize(window, &ww, &wh);
            glfwSetCursorPos(window, ww / 2.0, wh / 2.0);
        }
        // Don't let a key or button that's already down count as a fresh press in the menu.
        click_prev = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        enter_prev = glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_ENTER) == GLFW_PRESS;
        up_prev = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
        down_prev = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
        typed.clear();
        backspaces = 0;
        // A menu takes over the keyboard: an unsent chat line is dropped and the log closes.
        chat_open = false;
        chat_draft.clear();
        chat_log_open = false;
    }

    void close_menu() {
        screen = Screen::None;
        set_mouse_captured(true);
        fire_pressed = true; // the click that closed the menu must not also fire a shot
    }

    void handle_escape() {
        bool down = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
        if (down && !esc_pressed) {
            // In the game, Esc first backs out of the chat line, then the chat log, and only
            // then opens the menu.
            if (screen == Screen::None && chat_open) { chat_open = false; chat_draft.clear(); }
            else if (screen == Screen::None && chat_log_open) chat_log_open = false;
            else switch (screen) {
            case Screen::None: open_menu(Screen::Main); break;
            case Screen::Main: close_menu(); break;
            case Screen::Host: menu_action(MenuAction::CancelHost); break;
            case Screen::Join: menu_action(MenuAction::BackFromJoin); break;
            case Screen::Sound: menu_action(MenuAction::BackFromSound); break;
            case Screen::Controls: menu_action(MenuAction::BackFromControls); break;
            }
        }
        esc_pressed = down;
    }

    void menu_action(MenuAction a) {
        switch (a) {
        case MenuAction::Single:
            if (mode != Mode::Single) leave_multiplayer("");
            close_menu();
            break;
        case MenuAction::Join:
            if (mode != Mode::Single) leave_multiplayer("");
            menu_status.clear();
            menu_status_error = false;
            open_menu(Screen::Join);
            break;
        case MenuAction::Host:
            if (mode != Mode::Single) leave_multiplayer("");
            begin_hosting();
            break;
        case MenuAction::Exit:
            glfwSetWindowShouldClose(window, true);
            break;
        case MenuAction::Sound:
            // Just a settings page: it doesn't leave a multiplayer game.
            open_menu(Screen::Sound);
            break;
        case MenuAction::BackFromSound:
            if (volume_dirty) save_settings();
            open_menu(Screen::Main);
            menu_focus = 3; // back on the Sound button
            break;
        case MenuAction::Controls:
            // Just a page to read: it doesn't leave a multiplayer game either.
            open_menu(Screen::Controls);
            break;
        case MenuAction::BackFromControls:
            open_menu(Screen::Main);
            menu_focus = 4; // back on the Controls button
            break;
        case MenuAction::TestSound:
            play_sound("monster_sound.flac");
            break;
        case MenuAction::RequestStart: {
            net::Writer w;
            w.put<uint8_t>(MSG_REQUEST_START);
            net.send_to_host(w.buf, true);
            menu_status = "Starting...";
            menu_status_error = false;
            break;
        }
        case MenuAction::StartGame:
            host_start_round();
            break;
        case MenuAction::CancelHost:
        case MenuAction::BackFromJoin:
            leave_multiplayer("");
            open_menu(Screen::Main);
            break;
        case MenuAction::Connect:
            begin_join();
            break;
        default:
            break;
        }
    }

    float text_width(const std::string& text, float scale) const {
        float w = 0.0f;
        for (char c : text) if (c >= 32 && c < 127) w += cdata[c - 32].xadvance * scale;
        return w;
    }

    void draw_text_centered(const std::string& text, float cx, float y, glm::vec4 color, float scale = 1.0f) {
        draw_text(text, cx - text_width(text, scale) * 0.5f, y, color, scale);
    }

    // An immediate-mode menu: laid out, drawn and clicked in one pass each frame. The page is
    // described first and then scaled to fit, so every line and button shows in a small window
    // as well as on a big screen.
    void render_menu() {
        glViewport(0, 0, screen_width, screen_height);
        glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screen_width), static_cast<float>(screen_height), 0.0f, -1.0f, 1.0f);
        glm::mat4 id = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(id));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(id));
        glDisable(GL_DEPTH_TEST);

        const float W = static_cast<float>(screen_width), H = static_cast<float>(screen_height);
        const float cx = W * 0.5f;
        const glm::vec4 bright = { 1.0f, 1.0f, 1.0f, 1.0f }, dim = { 0.72f, 0.76f, 0.84f, 1.0f };
        const glm::vec4 good = { 0.50f, 1.00f, 0.60f, 1.0f }, bad = { 1.00f, 0.45f, 0.40f, 1.0f };
        const glm::vec4 gold = { 0.95f, 0.78f, 0.30f, 1.0f };
        const std::string port = std::to_string(net::DEFAULT_PORT);

        // ---- Describe the page -------------------------------------------------------------
        enum ItemKind { TEXT, INPUT, SLIDER, KEYROW, GAP };
        struct Item { ItemKind kind; std::string text; glm::vec4 color; float scale; float height; std::string text2; };
        std::vector<Item> items;
        auto text = [&](const std::string& s, glm::vec4 c) { items.push_back({ TEXT, s, c, 0.75f, 32.0f, "" }); };
        auto gap = [&](float h) { items.push_back({ GAP, "", {}, 0.0f, h, "" }); };
        // A key and what it does, in two columns.
        auto keyrow = [&](const std::string& key, const std::string& action) { items.push_back({ KEYROW, key, gold, 0.64f, 27.0f, action }); };
        items.push_back({ TEXT, "MAZE BEASTS", gold, 1.6f, 76.0f, "" });

        std::vector<Button> buttons;
        std::string hint;
        bool editable = false;

        if (screen == Screen::Main) {
            buttons = { { "Singleplayer", MenuAction::Single, true },
                        { "Multiplayer - Join", MenuAction::Join, true },
                        { "Multiplayer - Host", MenuAction::Host, true },
                        { "Sound", MenuAction::Sound, true },
                        { "Controls", MenuAction::Controls, true },
                        { "Exit", MenuAction::Exit, true } };
            hint = mode == Mode::Single ? "Esc: back to the game"
                                        : "Esc: back to the game.  Singleplayer, Join or Host leave this match.";
        }
        else if (screen == Screen::Controls) {
            text("Controls", bright);
            gap(4.0f);
            keyrow("W A S D", "Move");
            keyrow("Mouse", "Look and aim");
            keyrow("Left click", "Shoot");
            keyrow("Spacebar", "Jump");
            keyrow("Ctrl (hold)", "Crouch");
            keyrow("Tab", "Show the whole maze");
            keyrow("Esc", "Menu");
            keyrow("Enter", "Chat: type, then Enter again to send");
            keyrow("Y", "Chat log");
            keyrow("Ctrl+C / Ctrl+V", "Copy / paste (chat, and the Join address)");
            keyrow("F8", "New maze (singleplayer)");
            keyrow("F8 or chat: votemap", "Start a vote for a new maze (multiplayer)");
            keyrow("F1 / F2", "Vote Yes / No");
            keyrow("F5", "Developer mode (singleplayer only)");
            gap(6.0f);
            items.push_back({ TEXT, "Spectating: fly with W A S D and the mouse, Spacebar up, Ctrl down", dim, 0.6f, 28.0f, "" });
            buttons = { { "Back", MenuAction::BackFromControls, true } };
            hint = "Esc: back";
        }
        else if (screen == Screen::Sound) {
            for (int i = 0; i < left_presses; i++) set_volume(master_volume - 0.05f);
            for (int i = 0; i < right_presses; i++) set_volume(master_volume + 0.05f);
            text("Sound", bright);
            gap(8.0f);
            text("Volume: " + std::to_string(static_cast<int>(std::lround(master_volume * 100.0f))) + "%",
                 master_volume > 0.0f ? bright : dim);
            items.push_back({ SLIDER, "", bright, 0.0f, 64.0f });
            if (!sound_ready) text(opts.test ? "(sound is off in test mode)" : "(no audio device found)", dim);
            gap(8.0f);
            buttons = { { "Test sound", MenuAction::TestSound, sound_ready },
                        { "Back", MenuAction::BackFromSound, true } };
            hint = "Drag the slider or use Left/Right.  Esc: back";
        }
        else if (screen == Screen::Host) {
            text("Hosting on UDP port " + port, bright);
            text("Maze seed: " + std::to_string(round_seed), dim);
            std::string ips;
            for (const auto& ip : host_ips) ips += (ips.empty() ? "" : ", ") + ip;
            text("Other players join with: " + (ips.empty() ? std::string("your IP address") : ips), dim);
            text("(or 127.0.0.1 from another copy on this PC)", dim);
            gap(10.0f);
            int n = player_count();
            text(n <= 1 ? "Waiting for players..." : std::to_string(n) + "/3 players joined", n >= 2 ? good : bright);
            text("Player 1: you (host)", player_color(1));
            text((lobby_mask & (1 << 2)) ? "Player 2: joined" : "Player 2: waiting...",
                 (lobby_mask & (1 << 2)) ? player_color(2) : dim);
            text((lobby_mask & (1 << 3)) ? "Player 3: joined - will control a boss" : "Player 3 (optional): controls a boss",
                 (lobby_mask & (1 << 3)) ? player_color(3) : dim);
            buttons = { { "Start the game", MenuAction::StartGame, (lobby_mask & (1 << 2)) != 0 },
                        { "Cancel", MenuAction::CancelHost, true } };
            // Ctrl+C puts the address to share on the clipboard, ready to paste to the others.
            if (copy_presses && !host_ips.empty()) copy_to_clipboard(host_ips.front());
            hint = host_ips.empty() ? "Esc: stop hosting" : "Esc: stop hosting   Ctrl+C: copy your IP address";
        }
        else if (screen == Screen::Join) {
            editable = !join_connecting && !join_connected;
            if (editable) {
                // Only what can be part of an address gets in, typed or pasted, so a pasted
                // line's spaces or line break are simply left out.
                auto address_chars = [](const std::string& in) {
                    std::string out;
                    for (char c : in) {
                        bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                               || c == '.' || c == '-' || c == ':';
                        if (ok) out.push_back(c);
                    }
                    return out;
                };
                // Ctrl+V replaces the whole address: a pasted address is nearly always complete,
                // and there's no selecting text in this box to paste over.
                if (paste_presses) {
                    std::string pasted = address_chars(clipboard_text());
                    if (!pasted.empty()) join_address = pasted.substr(0, 64);
                }
                join_address += address_chars(typed).substr(0, 64 - std::min<size_t>(64, join_address.size()));
                for (int i = 0; i < backspaces && !join_address.empty(); i++) join_address.pop_back();
            }
            if (copy_presses && !join_address.empty()) copy_to_clipboard(join_address);
            text("Host or server IP address:", bright);
            items.push_back({ INPUT, join_address, bright, 0.9f, 60.0f });
            text("UDP port " + port + ", or type address:port for another", dim);
            gap(6.0f);
            bool can_start = false;
            if (join_connected) {
                std::string role = my_slot == 1 ? " - you start where singleplayer does"
                                 : my_slot == 2 ? " - you start at the exit" : " - you'll control a boss";
                text((server_dedicated ? "On the server as " : "Connected as ") + capitalised(player_name(my_slot)) + role,
                     player_color(my_slot));
                int n = player_count();
                std::string players = " (" + std::to_string(n) + "/3 players)";
                if (joined_in_progress) text("A maze is under way - you'll join in at the next one" + players, dim);
                else if (!server_dedicated) text("Waiting for the host to start the game" + players, dim);
                else if (my_slot != 1) text("Waiting for Player 1 to start the game" + players, dim);
                else if (n < 2) text("Waiting for another player to join" + players, dim);
                else { text("Start when everyone is here" + players, good); can_start = true; }
            }
            // On a dedicated server nobody is the host, so Player 1 starts the game from here.
            if (join_connected && server_dedicated && my_slot == 1)
                buttons = { { "Start the game", MenuAction::RequestStart, can_start },
                            { "Leave", MenuAction::BackFromJoin, true } };
            else
                buttons = { { "Connect", MenuAction::Connect, editable && !join_address.empty() },
                            { "Back", MenuAction::BackFromJoin, true } };
            hint = editable ? "Type the host's or server's IP address (or paste it: Ctrl+V), then press Enter" : "Esc: leave";
        }
        if (!menu_status.empty()) text(menu_status, menu_status_error ? bad : dim);
        gap(10.0f);
        if (note_showing()) hint = note; // "Copied: ..." for a moment

        // A first button that becomes available (Start, once someone joins) takes the focus,
        // so Enter starts the game instead of hitting the Cancel or Leave it was parked on.
        bool first_enabled = !buttons.empty() && buttons[0].enabled;
        if (screen == focus_screen && first_enabled && !first_was_enabled) menu_focus = 0;
        focus_screen = screen;
        first_was_enabled = first_enabled;

        // ---- Fit it to the window ----------------------------------------------------------
        const float button_h = 50.0f, button_gap = 12.0f, hint_h = hint.empty() ? 0.0f : 36.0f;
        float total = hint_h;
        for (const auto& it : items) total += it.height;
        total += buttons.size() * (button_h + button_gap);
        // Key rows line up in two columns: keys right-aligned against a gap, what they do after it.
        const float key_gap = 22.0f;
        float key_col = 0.0f, action_col = 0.0f;
        for (const auto& it : items) {
            if (it.kind != KEYROW) continue;
            key_col = std::max(key_col, text_width(it.text, it.scale));
            action_col = std::max(action_col, text_width(it.text2, it.scale));
        }
        float widest = key_col > 0.0f ? key_col + key_gap + action_col : 0.0f;
        for (const auto& it : items) if (it.kind == TEXT) widest = std::max(widest, text_width(it.text, it.scale));
        float ui = std::min((H - 24.0f) / total, (W - 32.0f) / std::max(widest, 440.0f));
        ui = std::clamp(ui, 0.35f, 1.3f);

        double mx, my;
        glfwGetCursorPos(window, &mx, &my);
        int ww, wh;
        glfwGetWindowSize(window, &ww, &wh);
        if (ww > 0 && wh > 0) { mx *= static_cast<double>(screen_width) / ww; my *= static_cast<double>(screen_height) / wh; }
        bool mouse_moved = mx != menu_last_mx || my != menu_last_my;
        menu_last_mx = mx;
        menu_last_my = my;
        bool click = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (!click) slider_dragging = false;

        // ---- Draw ------------------------------------------------------------------------------
        draw_quad(0.0f, 0.0f, W, H, { 0.02f, 0.02f, 0.04f, 0.80f });
        float y = std::max(8.0f, (H - total * ui) * 0.5f);
        for (const auto& it : items) {
            float h = it.height * ui;
            if (it.kind == TEXT && !it.text.empty()) {
                draw_text_centered(it.text, cx, y + h * 0.72f, it.color, it.scale * ui);
            }
            else if (it.kind == KEYROW) {
                float left = cx - (key_col + key_gap + action_col) * ui * 0.5f;
                float split = left + key_col * ui;
                float base = y + h * 0.72f;
                draw_text(it.text, split - text_width(it.text, it.scale * ui), base, it.color, it.scale * ui);
                draw_text(it.text2, split + key_gap * ui, base, bright, it.scale * ui);
            }
            else if (it.kind == INPUT) {
                float bw = std::min(460.0f * ui, W - 40.0f);
                float top = y + 4.0f * ui, bottom = top + 46.0f * ui;
                draw_quad(cx - bw / 2 - 2, top - 2, cx + bw / 2 + 2, bottom + 2, editable ? gold : dim);
                draw_quad(cx - bw / 2, top, cx + bw / 2, bottom, { 0.08f, 0.09f, 0.12f, 1.0f });
                bool caret = editable && std::fmod(glfwGetTime(), 1.0) < 0.5;
                draw_text_centered(it.text + (caret ? "_" : " "), cx, top + 33.0f * ui, it.color, it.scale * ui);
            }
            else if (it.kind == SLIDER) {
                // Grab anywhere along the track (or near it) and drag; the knob follows.
                float sw = std::min(420.0f * ui, W - 60.0f);
                float left = cx - sw / 2, right = cx + sw / 2, mid = y + h * 0.5f;
                bool over = mx >= left - 14.0f * ui && mx <= right + 14.0f * ui && my >= mid - 26.0f * ui && my <= mid + 26.0f * ui;
                if (click && !click_prev && over) slider_dragging = true;
                if (slider_dragging) set_volume(static_cast<float>((mx - left) / sw));
                float kx = left + sw * master_volume, th = 5.0f * ui;
                draw_quad(left, mid - th, right, mid + th, { 0.20f, 0.22f, 0.29f, 1.0f });
                draw_quad(left, mid - th, kx, mid + th, gold);
                draw_quad(kx - 9.0f * ui, mid - 18.0f * ui, kx + 9.0f * ui, mid + 18.0f * ui,
                          slider_dragging || over ? gold : bright);
            }
            y += h;
        }

        int count = static_cast<int>(buttons.size());
        auto enabled_at = [&](int i) { return i >= 0 && i < count && buttons[i].enabled; };
        menu_focus = std::clamp(menu_focus, 0, std::max(0, count - 1));
        if (!enabled_at(menu_focus)) for (int i = 0; i < count; i++) if (enabled_at(i)) { menu_focus = i; break; }

        const float bw = std::min(440.0f * ui, W - 40.0f), bh = button_h * ui;
        std::vector<glm::vec4> rects;
        for (int i = 0; i < count; i++) {
            glm::vec4 rc(cx - bw / 2, y, cx + bw / 2, y + bh);
            rects.push_back(rc);
            bool inside = mx >= rc.x && mx <= rc.z && my >= rc.y && my <= rc.w;
            if (inside && mouse_moved && buttons[i].enabled) menu_focus = i;
            bool focused = i == menu_focus && buttons[i].enabled;
            glm::vec4 bg = !buttons[i].enabled ? glm::vec4(0.13f, 0.13f, 0.16f, 0.85f)
                         : focused ? glm::vec4(0.95f, 0.78f, 0.30f, 0.97f) : glm::vec4(0.20f, 0.22f, 0.29f, 0.97f);
            glm::vec4 fg = !buttons[i].enabled ? glm::vec4(0.45f, 0.45f, 0.50f, 1.0f)
                         : focused ? glm::vec4(0.08f, 0.06f, 0.02f, 1.0f) : bright;
            draw_quad(rc.x, rc.y, rc.z, rc.w, bg);
            draw_text_centered(buttons[i].label, cx, rc.y + bh * 0.5f + 11.0f * ui, fg, ui);
            y += bh + button_gap * ui;
        }
        if (!hint.empty()) draw_text_centered(hint, cx, y + 22.0f * ui, dim, 0.6f * ui);
        const std::string version = "v0.42";
        const float vs = 0.55f * std::max(ui, 0.6f);
        draw_text(version, W - text_width(version, vs) - 12.0f, H - 10.0f, { 0.45f, 0.45f, 0.52f, 1.0f }, vs);
        glEnable(GL_DEPTH_TEST);

        // ---- Input, after drawing: an action may switch screens ---------------------------------
        MenuAction chosen = MenuAction::None;
        if (click && !click_prev && !slider_dragging) {
            for (int i = 0; i < count; i++) {
                const glm::vec4& rc = rects[i];
                if (buttons[i].enabled && mx >= rc.x && mx <= rc.z && my >= rc.y && my <= rc.w) chosen = buttons[i].action;
            }
        }
        click_prev = click;
        bool up = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
        bool down = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
        bool enter = glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_ENTER) == GLFW_PRESS;
        if (up && !up_prev) for (int i = menu_focus - 1; i >= 0; i--) if (enabled_at(i)) { menu_focus = i; break; }
        if (down && !down_prev) for (int i = menu_focus + 1; i < count; i++) if (enabled_at(i)) { menu_focus = i; break; }
        if (enter && !enter_prev && enabled_at(menu_focus)) chosen = buttons[menu_focus].action;
        up_prev = up;
        down_prev = down;
        enter_prev = enter;

        if (chosen != MenuAction::None) menu_action(chosen);
    }

    // Names over the other players' heads, only when you could actually see them.
    void render_labels() {
        if (mode == Mode::Single || !in_round || tab_view) return;
        glViewport(0, 0, screen_width, screen_height);
        glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screen_width), static_cast<float>(screen_height), 0.0f, -1.0f, 1.0f);
        glm::mat4 id = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(id));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(id));
        glDisable(GL_DEPTH_TEST);

        struct Tag { glm::vec3 pos; std::string name; glm::vec4 color; int level; };
        std::vector<Tag> tags;
        for (int s = 1; s <= 2; s++) {
            if (s == my_slot) continue;
            const RemotePlayer& r = remote[s];
            if (!r.present || !r.has_state || !r.alive) continue;
            float head = static_cast<float>(floor_height(r.rx, r.ry, r.level) + r.jump + 0.68 - crouch_height_drop(r.crouch)); // just over the helmet
            tags.push_back({ glm::vec3(static_cast<float>(r.rx), head, static_cast<float>(r.ry)), player_name(s), player_color(s), r.level });
        }
        if (beast_active() && my_slot != 3) {
            if (const Monster* m = find_monster(beast_boss_id))
                tags.push_back({ glm::vec3(static_cast<float>(m->x), 1.0f, static_cast<float>(m->y)), "Player 3 - the Beast", player_color(3), 0 });
        }
        for (const auto& t : tags) {
            if (t.level != player_level) continue;
            if (std::hypot(t.pos.x - player_pos_x, t.pos.z - player_pos_y) > 14.0) continue;
            if (!line_of_sight(t.level, player_pos_x, player_pos_y, t.pos.x, t.pos.z)) continue;
            glm::vec4 clip = last_proj * last_view * glm::vec4(t.pos, 1.0f);
            if (clip.w < 0.05f) continue; // behind the camera
            glm::vec3 ndc = glm::vec3(clip) / clip.w;
            if (std::fabs(ndc.x) > 1.05f || std::fabs(ndc.y) > 1.05f) continue;
            float sx = (ndc.x * 0.5f + 0.5f) * screen_width;
            float sy = (1.0f - (ndc.y * 0.5f + 0.5f)) * screen_height;
            draw_text_centered(t.name, sx, sy, t.color, 0.6f);
        }
        glEnable(GL_DEPTH_TEST);
    }

    // Big centre-screen messages: winning, dying, locked exits, the end of a round.
    void render_center_message(int bosses) {
        glDisable(GL_DEPTH_TEST);
        float cx = screen_width * 0.5f, cy = screen_height * 0.5f;
        if (mode == Mode::Single) {
            if (showing_win) draw_text_centered("You Won!", cx, cy, { 1, 1, 0, 1 });
            else if (showing_die) draw_text_centered("You Died!", cx, cy, { 1, 0, 0, 1 });
        }
        else if (in_round && round_over) {
            std::string who = round_winner == my_slot ? "You escaped the maze!" : capitalised(player_name(round_winner)) + " escaped the maze!";
            draw_text_centered(who, cx, cy, round_winner == my_slot ? glm::vec4(1, 1, 0, 1) : player_color(round_winner));
            int secs = std::max(0, static_cast<int>(std::ceil(round_over_until - glfwGetTime())));
            draw_text_centered("Next maze in " + std::to_string(secs) + "...", cx, cy + 40.0f, { 0.85f, 0.85f, 0.9f, 1.0f }, 0.8f);
        }
        else if (showing_die) {
            // Above the middle, out of the way of the view, but below the kill feed.
            float top = screen_height * 0.30f;
            const glm::vec4 hint_col = { 0.80f, 0.82f, 0.88f, 1.0f };
            draw_text_centered(killed_by == 0 ? std::string("You were killed by a monster")
                                              : "You were killed by " + player_name(killed_by), cx, top, { 1, 0.2f, 0.2f, 1 });
            int secs = std::max(1, static_cast<int>(std::ceil(die_timer - glfwGetTime())));
            draw_text_centered("Respawning in " + std::to_string(secs) + "...", cx, top + 40.0f, { 1, 1, 1, 1 }, 0.8f);
            draw_text_centered("Spectating: fly with WASD and the mouse, Space up, Ctrl down", cx, top + 72.0f, hint_col, 0.6f);
        }
        else if (spectating) {
            float top = screen_height * 0.30f;
            draw_text_centered("Your beast was defeated - spectating until the next maze", cx, top, player_color(3), 0.8f);
            draw_text_centered("Fly with WASD and the mouse, Space up, Ctrl down", cx, top + 34.0f, { 0.80f, 0.82f, 0.88f, 1.0f }, 0.6f);
        }
        // Only in a real game: in a lobby the paused maze behind the menu isn't yours to exit.
        bool playing = mode == Mode::Single || in_round;
        if (playing && !showing_die && !showing_win && !(in_round && round_over) && !is_beast()) {
            // Standing on your exit with bosses still alive: tell the player it's locked.
            auto e = exit_cell(my_slot);
            bool at_exit = player_level == 0 && static_cast<int>(player_pos_x) == e.first && static_cast<int>(player_pos_y) == e.second;
            if (at_exit && bosses > 0) {
                std::string msg = "Exit locked: " + std::to_string(bosses) + (bosses == 1 ? " boss remaining" : " bosses remaining");
                draw_text_centered(msg, cx, cy, { 1.0f, 0.3f, 0.3f, 1.0f });
            }
        }
        glEnable(GL_DEPTH_TEST);
    }

    // ---- The vote panel -------------------------------------------------------------------------
    // Under the minimap on the left: what's being voted on, the countdown, how to vote, and
    // each player's answer so far. Once decided, the result stays up for a few seconds.
    void render_vote() {
        if (mode == Mode::Single || !in_round || screen != Screen::None || !vote_view.shown) return;
        const double now = glfwGetTime();
        const VoteView& v = vote_view;
        bool open = v.state == VOTE_OPEN;
        if (!open && now >= v.hide_at) return;

        glViewport(0, 0, screen_width, screen_height);
        glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screen_width), static_cast<float>(screen_height), 0.0f, -1.0f, 1.0f);
        glm::mat4 id = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(id));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(id));
        glDisable(GL_DEPTH_TEST);

        const glm::vec4 white = { 1.0f, 1.0f, 1.0f, 1.0f }, dim = { 0.70f, 0.74f, 0.82f, 1.0f };
        const glm::vec4 gold = { 0.95f, 0.78f, 0.30f, 1.0f }, green = { 0.45f, 1.0f, 0.55f, 1.0f }, red = { 1.0f, 0.45f, 0.40f, 1.0f };
        struct Line { std::string text; glm::vec4 color; std::string right; };
        std::vector<Line> lines;
        uint8_t me = static_cast<uint8_t>(1 << my_slot);
        if (open) {
            int secs = std::max(0, static_cast<int>(std::ceil(v.ends_at - now)));
            lines.push_back({ "New maze?", gold, std::to_string(secs) + "s" });
            lines.push_back({ "Asked by " + player_name(v.starter), dim, "" });
            if (!(v.voters & me)) lines.push_back({ "(you'll vote from the next maze)", dim, "" });
            else if (v.yes & me) lines.push_back({ "You voted Yes", green, "" });
            else if (v.no & me) lines.push_back({ "You voted No", red, "" });
            else lines.push_back({ "F1 = Yes      F2 = No", white, "" });
            for (int s = 1; s <= 3; s++) {
                uint8_t b = static_cast<uint8_t>(1 << s);
                if (!(v.voters & b)) continue;
                bool y = (v.yes & b) != 0, n = (v.no & b) != 0;
                lines.push_back({ capitalised(player_name(s)) + ":", player_color(s), y ? "Yes" : n ? "No" : "..." });
            }
        }
        else {
            lines.push_back({ v.state == VOTE_PASSED ? "Vote passed: new maze!" : v.state == VOTE_FAILED ? "Vote failed" : "Vote expired",
                              v.state == VOTE_PASSED ? green : v.state == VOTE_FAILED ? red : dim, "" });
        }

        const float s = chat_scale() * 1.05f, line_h = 34.0f * s, pad = 10.0f * s, ascent = 27.0f * s;
        const int mini = std::max(64, std::min(screen_width, screen_height) / 5); // as render_minimap sizes it
        const float x = 16.0f, y = 16.0f + mini + 12.0f;
        float w = 0.0f;
        for (const auto& l : lines) w = std::max(w, text_width(l.text, s) + (l.right.empty() ? 0.0f : 28.0f * s + text_width(l.right, s)));
        w = std::max(w + 2.0f * pad, 200.0f * s);
        float h = lines.size() * line_h + 2.0f * pad;
        draw_quad(x - 2.0f, y - 2.0f, x + w + 2.0f, y + h + 2.0f, open ? gold : faded(gold, 0.5f));
        draw_quad(x, y, x + w, y + h, { 0.04f, 0.05f, 0.08f, 0.88f });
        for (size_t i = 0; i < lines.size(); i++) {
            const Line& l = lines[i];
            float base = y + pad + i * line_h + ascent;
            draw_text(l.text, x + pad, base, l.color, s);
            if (!l.right.empty()) {
                glm::vec4 rc = l.right == "Yes" ? green : l.right == "No" ? red : i == 0 ? white : dim;
                draw_text(l.right, x + w - pad - text_width(l.right, s), base, rc, s);
            }
        }
        glEnable(GL_DEPTH_TEST);
    }

    // ---- Chat on screen ----------------------------------------------------------------------

    float chat_scale() const { return std::clamp(screen_height / 1150.0f, 0.46f, 0.8f); }

    // The chat column is centred and, on a wide enough window, clear of the health bar on the
    // left.
    float chat_column_width() const {
        float W = static_cast<float>(screen_width);
        float w = std::min(W * 0.56f, W - 760.0f);
        return std::max(w, std::min(W - 24.0f, 320.0f));
    }

    // Splits text into lines no wider than max_w, breaking at spaces where it can and inside a
    // word only when one word alone is too long.
    std::vector<std::string> wrap_text(const std::string& text, float scale, float max_w) const {
        std::vector<std::string> lines;
        std::string line;
        float width = 0.0f;
        size_t space = std::string::npos; // the last space in `line`
        for (char c : text) {
            if (c < 32 || c > 126) continue;
            float cw = cdata[c - 32].xadvance * scale;
            if (width + cw > max_w && !line.empty()) {
                if (c == ' ') { // break right here, dropping the space
                    lines.push_back(line);
                    line.clear();
                    width = 0.0f;
                    space = std::string::npos;
                    continue;
                }
                std::string rest;
                if (space != std::string::npos) {
                    rest = line.substr(space + 1);
                    line.resize(space);
                }
                lines.push_back(line);
                line = rest;
                width = text_width(line, scale);
                space = std::string::npos;
            }
            if (c == ' ' && line.empty() && !lines.empty()) continue; // no space at the start of a wrapped line
            line.push_back(c);
            width += cw;
            if (c == ' ') space = line.size() - 1;
        }
        if (!line.empty() || lines.empty()) lines.push_back(line);
        return lines;
    }

    // Draws a line whose start is in other colours: each part is (character count, colour),
    // and whatever follows them is drawn in `rest`.
    void draw_segments(const std::string& line, float x, float y, float scale,
                       std::initializer_list<std::pair<size_t, glm::vec4>> parts, glm::vec4 rest) {
        size_t pos = 0;
        for (const auto& part : parts) {
            if (pos >= line.size()) return;
            std::string piece = line.substr(pos, part.first);
            draw_text(piece, x, y, part.second, scale);
            x += text_width(piece, scale);
            pos += piece.size();
        }
        if (pos < line.size()) draw_text(line.substr(pos), x, y, rest, scale);
    }

    static glm::vec4 faded(glm::vec4 c, float alpha) { return { c.r, c.g, c.b, c.a * alpha }; }

    // The chat along the bottom centre (the line being typed, with recent messages stacked
    // above it), plus the Y log window when it's open.
    void render_chat() {
        if (screen != Screen::None) return;
        const double now = glfwGetTime();
        bool recent_any = !chat_history.empty() && chat_history.back().shown_until > now;
        if (!chat_open && !chat_log_open && !recent_any) return;

        glViewport(0, 0, screen_width, screen_height);
        glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screen_width), static_cast<float>(screen_height), 0.0f, -1.0f, 1.0f);
        glm::mat4 id = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(id));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(id));
        glDisable(GL_DEPTH_TEST);

        const float W = static_cast<float>(screen_width), H = static_cast<float>(screen_height);
        const float s = chat_scale(), line_h = 38.0f * s, pad = 8.0f * s, ascent = 30.0f * s;
        const float col_w = chat_column_width(), cx = W * 0.5f;
        const glm::vec4 white = { 1.0f, 1.0f, 1.0f, 1.0f }, dim = { 0.72f, 0.76f, 0.84f, 1.0f };
        const glm::vec4 gold = { 0.95f, 0.78f, 0.30f, 1.0f }, red = { 1.0f, 0.45f, 0.40f, 1.0f };
        const float bottom = H - 14.0f;

        // The line being typed: a box along the bottom that grows upward as the text wraps.
        // Its space is kept even while it's closed, so messages don't jump when it opens.
        float input_h = line_h + 2.0f * pad;
        if (chat_open) {
            bool caret = std::fmod(now, 1.0) < 0.5;
            auto lines = wrap_text("Say: " + chat_draft + (caret ? "_" : " "), s, col_w - 2.0f * pad);
            input_h = lines.size() * line_h + 2.0f * pad;
            float top = bottom - input_h, left = cx - col_w * 0.5f, right = cx + col_w * 0.5f;
            draw_quad(left - 2.0f, top - 2.0f, right + 2.0f, bottom + 2.0f, gold);
            draw_quad(left, top, right, bottom, { 0.05f, 0.06f, 0.09f, 0.92f });
            for (size_t i = 0; i < lines.size(); i++) {
                float y = top + pad + line_h * i + ascent;
                if (i == 0) draw_segments(lines[i], left + pad, y, s, { { 5, gold } }, white);
                else draw_text(lines[i], left + pad, y, white, s);
            }
            // Just above the box: how to finish, and how much room is left.
            const float hs = s * 0.8f;
            std::string hint = note_showing() ? note : "Enter: send   Esc: cancel   Ctrl+V: paste   Ctrl+C: copy";
            std::string count = std::to_string(chat_draft.size()) + "/" + std::to_string(CHAT_MAX_CHARS);
            draw_text(hint, left, top - 7.0f * s, dim, hs);
            draw_text(count, right - text_width(count, hs), top - 7.0f * s, chat_draft.size() >= CHAT_MAX_CHARS ? red : dim, hs);
            input_h += 26.0f * s; // and room for that line
        }

        // Recent messages, newest at the bottom: the last CHAT_ON_SCREEN that are still within
        // their CHAT_SHOW_SECONDS, each fading out over its final second. A ninth pushes the
        // oldest off the top. They stop short of the middle of the screen.
        std::vector<int> recent;
        for (int i = static_cast<int>(chat_history.size()) - 1; i >= 0 && static_cast<int>(recent.size()) < CHAT_ON_SCREEN; i--) {
            if (chat_history[i].shown_until <= now) break; // anything older has gone as well
            recent.push_back(i);
        }
        float y = bottom - input_h - 6.0f * s; // bottom edge of the next line to draw, going up
        const float top_limit = H * 0.42f;
        bool full = false;
        for (int idx : recent) {
            const ChatMessage& m = chat_history[idx];
            float alpha = static_cast<float>(std::clamp(m.shown_until - now, 0.0, 1.0));
            auto lines = wrap_text(m.sender + ": " + m.text, s, col_w - 2.0f * pad);
            for (int li = static_cast<int>(lines.size()) - 1; li >= 0; li--) {
                if (y - line_h < top_limit) { full = true; break; }
                float w = text_width(lines[li], s), left = cx - w * 0.5f;
                draw_quad(left - pad, y - line_h, left + w + pad, y, { 0.0f, 0.0f, 0.0f, 0.55f * alpha });
                float base = y - line_h + ascent + 2.0f * s;
                if (li == 0) draw_segments(lines[li], left, base, s, { { m.sender.size() + 1, faded(player_color_of_sender(m), alpha) } }, faded(white, alpha));
                else draw_text(lines[li], left, base, faded(white, alpha), s);
                y -= line_h;
            }
            if (full) break;
            y -= 3.0f * s; // a little space between messages
        }

        if (chat_log_open) render_chat_log();
        glEnable(GL_DEPTH_TEST);
    }

    glm::vec4 player_color_of_sender(const ChatMessage& m) const {
        if (m.from < 1 || m.from > 3) return { 0.95f, 0.78f, 0.30f, 1.0f }; // the game's own notices, in gold
        return mode != Mode::Single ? player_color(m.from) : glm::vec4(0.40f, 1.00f, 0.50f, 1.0f);
    }

    // The Y window: everything said since the game started, newest at the bottom. Up/Down
    // pick a message (Ctrl+C copies it); the wheel, Page Up/Down, Home and End scroll.
    void render_chat_log() {
        const float W = static_cast<float>(screen_width), H = static_cast<float>(screen_height);
        const float s = chat_scale() * 0.92f, line_h = 36.0f * s, ascent = 28.0f * s;
        const float pad = 10.0f * s, title_h = 42.0f * s, hint_h = 32.0f * s, bar_w = 7.0f * s;
        const glm::vec4 white = { 1.0f, 1.0f, 1.0f, 1.0f }, dim = { 0.66f, 0.70f, 0.78f, 1.0f };
        const glm::vec4 gold = { 0.95f, 0.78f, 0.30f, 1.0f };

        const float pw = std::clamp(W * 0.5f, std::min(W - 20.0f, 380.0f), 900.0f);
        const float ph = std::clamp(H * 0.48f, std::min(H - 20.0f, 220.0f), 640.0f);
        const float px = (W - pw) * 0.5f;
        const float py = std::clamp(std::max(H * 0.1f, 50.0f), 10.0f, std::max(10.0f, H - 10.0f - ph)); // below the objective line
        draw_quad(px - 2.0f, py - 2.0f, px + pw + 2.0f, py + ph + 2.0f, { 0.95f, 0.78f, 0.30f, 0.85f });
        draw_quad(px, py, px + pw, py + ph, { 0.04f, 0.05f, 0.08f, 0.92f });

        const int n = static_cast<int>(chat_history.size());
        draw_text("Chat log (" + std::to_string(n) + (n == 1 ? " message)" : " messages)"), px + pad, py + title_h * 0.7f, gold, s * 1.05f);

        // Wrap the history to this width (again only if the window changed size; otherwise
        // just the messages that are new since last time).
        const float x0 = px + pad, text_w = pw - 3.0f * pad - bar_w;
        const float y0 = py + title_h, y1 = py + ph - hint_h;
        if (text_w != chat_log_wrap_w || s != chat_log_wrap_scale) {
            chat_log_lines.clear();
            chat_log_first_line.clear();
            chat_log_wrap_w = text_w;
            chat_log_wrap_scale = s;
        }
        for (size_t mi = chat_log_first_line.size(); mi < chat_history.size(); mi++) {
            const ChatMessage& m = chat_history[mi];
            chat_log_first_line.push_back(static_cast<int>(chat_log_lines.size()));
            for (auto& l : wrap_text("[" + m.stamp + "] " + m.sender + ": " + m.text, s, text_w))
                chat_log_lines.push_back({ static_cast<int>(mi), l });
        }

        // Keys and wheel.
        const int total = static_cast<int>(chat_log_lines.size());
        const int visible = std::max(1, static_cast<int>((y1 - y0) / line_h));
        const int max_top = std::max(0, total - visible);
        bool scrolled = false;
        int scroll = -static_cast<int>(std::lround(wheel * 3.0));
        scroll += (page_down_presses - page_up_presses) * std::max(1, visible - 1);
        if (scroll != 0) { chat_log_top += scroll; scrolled = true; }
        if (home_presses) { chat_log_top = 0; scrolled = true; }
        if (end_presses) { chat_log_top = max_top; scrolled = true; }
        if (n > 0 && (up_presses || down_presses)) {
            int sel = chat_log_selected < 0 || chat_log_selected >= n ? n - 1 : chat_log_selected;
            chat_log_selected = std::clamp(sel - up_presses + down_presses, 0, n - 1);
            // Bring the picked message into view.
            int first = chat_log_first_line[chat_log_selected];
            int last = (chat_log_selected + 1 < n ? chat_log_first_line[chat_log_selected + 1] : total) - 1;
            if (first < chat_log_top) chat_log_top = first;
            if (last >= chat_log_top + visible) chat_log_top = last - visible + 1;
            scrolled = true;
        }
        if (scrolled) chat_log_follow = chat_log_top >= max_top;
        if (chat_log_follow) chat_log_top = max_top;
        chat_log_top = std::clamp(chat_log_top, 0, max_top);

        if (n == 0) {
            draw_text_centered("Nothing has been said yet. Press Enter to chat.", px + pw * 0.5f, (y0 + y1) * 0.5f, dim, s);
        }
        for (int i = chat_log_top; i < std::min(total, chat_log_top + visible); i++) {
            const LogLine& l = chat_log_lines[i];
            const ChatMessage& m = chat_history[l.msg];
            float top = y0 + (i - chat_log_top) * line_h;
            if (l.msg == chat_log_selected)
                draw_quad(px + 4.0f, top, px + pw - 2.0f * pad - bar_w, top + line_h, { 0.22f, 0.25f, 0.34f, 0.95f });
            float base = top + ascent;
            if (chat_log_first_line[l.msg] == i)
                draw_segments(l.text, x0, base, s, { { m.stamp.size() + 3, dim }, { m.sender.size() + 1, player_color_of_sender(m) } }, white);
            else
                draw_text(l.text, x0, base, white, s);
        }

        // Scrollbar, when there's more than fits.
        if (total > visible) {
            float bx = px + pw - pad - bar_w;
            draw_quad(bx, y0, bx + bar_w, y1, { 0.18f, 0.20f, 0.26f, 1.0f });
            float thumb = std::max(18.0f * s, (y1 - y0) * visible / total);
            float t = max_top > 0 ? static_cast<float>(chat_log_top) / max_top : 1.0f;
            float ty = y0 + (y1 - y0 - thumb) * t;
            draw_quad(bx, ty, bx + bar_w, ty + thumb, gold);
        }

        std::string hint = note_showing() ? note : "Up/Down: pick   Ctrl+C: copy   Wheel, PgUp/PgDn: scroll   Y: close";
        float hs = s * 0.85f;
        hs = std::min(hs, (pw - 2.0f * pad) / std::max(1.0f, text_width(hint, 1.0f)));
        draw_text(hint, px + pad, py + ph - hint_h * 0.32f, note_showing() ? gold : dim, hs);
    }

    void run() {
        last_time = glfwGetTime();
        const double launch_time = last_time;
        double last_mouse_x = screen_width / 2.0;
        double last_mouse_y = screen_height / 2.0;
        while (!glfwWindowShouldClose(window)) {
            double current_time = glfwGetTime();
            // Clamped, so a stall (dragging the window, a debugger pause) can't fling things
            // through walls in one giant step.
            double delta = std::min(current_time - last_time, 0.25);
            last_time = current_time;
            if (opts.quit_after > 0.0 && current_time - launch_time > opts.quit_after)
                glfwSetWindowShouldClose(window, true);

            handle_escape();
            pump_network();
            update_chat(delta);
            if (mode == Mode::Host && in_round) host_settle_vote(); // votes expire on the clock
            if (mode == Mode::Host && !in_round && opts.autostart > 0 && (lobby_mask & (1 << 2))
                && player_count() >= opts.autostart)
                host_start_round();

            // Singleplayer pauses behind the menu. A multiplayer game can't pause for one player,
            // so it carries on; you just stand still while your menu is open.
            bool world_running = mode == Mode::Single ? screen == Screen::None : in_round;

            if (world_running) {
                process_input(delta, last_mouse_x, last_mouse_y);
                if (opts.test_fire && mode != Mode::Single) {
                    test_fire_timer -= delta;
                    if (test_fire_timer <= 0.0) {
                        test_fire_timer = 2.0;
                        if (is_beast()) beast_shoot();
                        else if (!showing_die) shoot();
                    }
                }

                update_projectiles(delta);
                update_monster_projectiles(delta);
                if (mode == Mode::Client) smooth_monsters(delta); // the host runs the monsters
                else update_monsters(delta);
                smooth_remote_players(delta);
                update_audio_cues();

                // Monsters and health packs sit on the maze level; standing under them doesn't count.
                if (damage_cooldown > 0) damage_cooldown--;
                else if (player_level == 0 && !is_beast()) {
                    for (auto& m : monsters) {
                        double dist = std::hypot(m.x - player_pos_x, m.y - player_pos_y);
                        if (dist < 0.8) {
                            take_damage(1, (beast_active() && m.id == beast_boss_id) ? 3 : 0);
                            damage_cooldown = 30;
                            break;
                        }
                    }
                }

                if (!is_beast() && !showing_die) { // a spectating ghost can't pick anything up
                    for (auto it = health_packs.begin(); it != health_packs.end(); ) {
                        if (it->level == player_level && std::hypot(it->x - player_pos_x, it->y - player_pos_y) < MEDPACK_PICKUP_RADIUS) {
                            player_hp = max_hp;
                            int id = it->id;
                            it = health_packs.erase(it);
                            // The host decides who really got it; a tie just heals you both.
                            net::Writer w;
                            if (mode == Mode::Client) {
                                w.put<uint8_t>(MSG_PICKUP).put<uint8_t>(round_id).put<int32_t>(id);
                                net.send_to_host(w.buf, true);
                            }
                            else if (mode == Mode::Host) {
                                w.put<uint8_t>(MSG_PACK_GONE).put<uint8_t>(round_id).put<int32_t>(id);
                                net.broadcast(w.buf, true);
                            }
                        }
                        else {
                            ++it;
                        }
                    }
                }

                int bosses = boss_count();
                auto goal = exit_cell(my_slot);
                bool at_exit = !is_beast() && player_level == 0
                    && static_cast<int>(player_pos_x) == goal.first && static_cast<int>(player_pos_y) == goal.second;
                if (mode == Mode::Single) {
                    if (!showing_win && at_exit && bosses == 0) {
                        showing_win = true;
                        win_timer = current_time + 1.0;
                    }
                }
                else if (!round_over && !showing_die) {
                    if (at_exit && bosses == 0) {
                        if (mode == Mode::Host) end_round(my_slot);
                        else if (!exit_reported) {
                            net::Writer w;
                            w.put<uint8_t>(MSG_REACHED_EXIT).put<uint8_t>(round_id).put<uint8_t>(static_cast<uint8_t>(my_slot));
                            net.send_to_host(w.buf, true);
                            exit_reported = true;
                        }
                    }
                    if (!at_exit) exit_reported = false; // try again if the host said "not yet"
                }

                if (!showing_die && player_hp <= 0 && !is_beast()) {
                    showing_die = true;
                    killed_by = last_damage_by;
                    if (mode == Mode::Single) {
                        die_timer = current_time + 1.5;
                    }
                    else {
                        // Multiplayer: a few seconds as a free camera before respawning.
                        die_timer = current_time + RESPAWN_SECONDS;
                        start_free_fly();
                        net::Writer w;
                        w.put<uint8_t>(MSG_DEATH).put<uint8_t>(round_id).put<uint8_t>(static_cast<uint8_t>(my_slot))
                         .put<uint8_t>(static_cast<uint8_t>(killed_by));
                        send_msg(w.buf, true);
                        add_feed(death_text(my_slot, killed_by));
                    }
                }

                if (showing_win && current_time > win_timer) {
                    regenerate_maze();
                }
                else if (showing_die && current_time > die_timer) {
                    if (mode == Mode::Single) respawn_player(); // same maze, back to the start
                    else { place_player_at_spawn(true); showing_die = false; } // your own start
                }

                if (mode != Mode::Single) send_periodic(delta);
            }
            else if (mode == Mode::Single) {
                stop_boss_sound(); // paused behind the menu
            }

            if (mode == Mode::Host && in_round && round_over && current_time > round_over_until)
                host_start_round();
            feed.erase(std::remove_if(feed.begin(), feed.end(),
                       [current_time](const std::pair<std::string, double>& f) { return f.second < current_time; }), feed.end());

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
                // The Beast is a boss, not someone holding a gun; a spectator has no body at all.
                if (!is_beast() && !free_fly) render_viewmodel();
                render_minimap();
            }

            int hud_bosses = boss_count();
            render_hud(hud_bosses);
            render_labels();
            render_center_message(hud_bosses);
            render_vote();
            render_chat();
            if (screen != Screen::None) render_menu();

            // This frame's key presses have all been used (or ignored) by now.
            typed.clear();
            backspaces = 0;
            left_presses = right_presses = 0;
            enter_presses = y_presses = copy_presses = paste_presses = 0;
            f1_presses = f2_presses = 0;
            up_presses = down_presses = page_up_presses = page_down_presses = 0;
            home_presses = end_presses = 0;
            wheel = 0.0;
            if (net.active()) net.flush();

            if (!opts.screenshot.empty() && glfwWindowShouldClose(window)) save_screenshot(opts.screenshot);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void process_input(double delta, double& last_mouse_x, double& last_mouse_y) {
        // Esc is handled in the main loop: it opens the menu. With a menu open (only possible
        // here during a multiplayer round) the player just stands still; physics carries on.
        // While a chat line is open the keyboard types into it, so the game's keys do nothing,
        // though the mouse still looks around.
        bool controls = screen == Screen::None;
        bool keys = controls && !chat_open;
        bool can_move = keys && !spectating;

        // F8 makes a new maze: in singleplayer straight away; in multiplayer it starts a vote
        // for one (as typing "votemap" does). F1 and F2 vote Yes and No. One press, one action.
        bool f8_down = keys && glfwGetKey(window, GLFW_KEY_F8) == GLFW_PRESS;
        if (f8_down && !f8_pressed) {
            if (mode == Mode::Single) regenerate_maze();
            else request_vote();
        }
        f8_pressed = f8_down;
        if (keys && mode != Mode::Single) {
            if (f1_presses) cast_vote(true);
            else if (f2_presses) cast_vote(false);
        }

        // Crouch while Ctrl is held: nobody crouches while typing, spectating (Ctrl flies down
        // then), dead, or as the Beast. The change runs over CROUCH_SECONDS either way.
        crouching = keys && !free_fly && !showing_die && !is_beast()
                 && (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
        double crouch_step = delta / CROUCH_SECONDS;
        crouch = crouching ? std::min(1.0, crouch + crouch_step) : std::max(0.0, crouch - crouch_step);

        // Tab toggles the map view (press to switch, no longer hold-to-view).
        bool tab_down = keys && glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
        if (tab_down && !tab_pressed) tab_view = !tab_view;
        tab_pressed = tab_down;

        // F5 toggles developer mode; F6 teleports through boss rooms while it is enabled.
        // Singleplayer only - teleporting around a shared maze would just be cheating.
        bool dev_keys = keys && mode == Mode::Single;
        bool f5_down = dev_keys && glfwGetKey(window, GLFW_KEY_F5) == GLFW_PRESS;
        if (f5_down && !f5_pressed) dev_mode = !dev_mode;
        f5_pressed = f5_down;

        bool f6_down = dev_keys && glfwGetKey(window, GLFW_KEY_F6) == GLFW_PRESS;
        if (f6_down && !f6_pressed && dev_mode) dev_teleport_to_boss_room();
        f6_pressed = f6_down;

        bool f7_down = dev_keys && glfwGetKey(window, GLFW_KEY_F7) == GLFW_PRESS;
        if (f7_down && !f7_pressed && dev_mode) dev_teleport_near_exit();
        f7_pressed = f7_down;

        double move_speed = 0.04182 * 60.0 * delta; // 15% slower again (was 0.0492)
        if (is_beast()) move_speed *= BEAST_SPEED;
        move_speed *= 1.0 - (1.0 - CROUCH_SPEED) * eased(crouch);

        if (free_fly) {
            if (keys) fly(move_speed * 1.5);
        }
        else {
            if (!can_move) move_speed = 0.0;

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
            // Bosses don't hop, so neither does the Beast.
            if (grounded && can_move && !is_beast() && glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) jump_velocity = JUMP_SPEED;
            if (!grounded || jump_velocity > 0.0) {
                jump_velocity -= GRAVITY * delta;
                jump_height += jump_velocity * delta;
                if (jump_height <= 0.0) { jump_height = 0.0; jump_velocity = 0.0; }
            }
        }

        recoil = std::max(0.0, recoil - delta);
        // Menu open: no looking around or shooting. Nor in a window that isn't in front: with
        // two copies side by side the cursor keeps crossing the other one, and that mustn't
        // swing its view around.
        bool focused = glfwGetWindowAttrib(window, GLFW_FOCUSED) != 0;
        if (!controls || !focused) {
            first_mouse = true;
            fire_pressed = true; // and the click that brings it back to the front won't fire
            return;
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
        bool fire_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (fire_down && !fire_pressed && can_move && !showing_die && !free_fly) {
            if (is_beast()) beast_shoot();
            else {
                shoot();
                recoil = recoil_time;
            }
        }
        fire_pressed = fire_down;
    }

    void render_3d() {
        glViewport(0, 0, screen_width, screen_height);
        glm::mat4 projection = glm::perspective(glm::radians(60.0f), static_cast<float>(screen_width) / screen_height, 0.01f, 100.0f);  // no wall see-through when close
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glm::vec3 camera_pos(static_cast<float>(player_pos_x), static_cast<float>(camera_y()), static_cast<float>(player_pos_y));
        float cp = std::cos(glm::radians(static_cast<float>(pitch)));
        float sp = std::sin(glm::radians(static_cast<float>(pitch)));
        glm::vec3 camera_front(static_cast<float>(dir_x) * cp, sp, static_cast<float>(dir_y) * cp);
        glm::vec3 camera_up(0.0f, 1.0f, 0.0f);
        glm::mat4 view = glm::lookAt(camera_pos, camera_pos + camera_front, camera_up);
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));
        last_proj = projection; // kept for placing name labels over other players
        last_view = view;

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

        // Exit marker: a red square on the floor of your exit cell (model is still identity).
        // Player 2's exit is Player 1's start; the Beast has none.
        if (!is_beast()) {
            auto goal = exit_cell(my_slot);
            float ex = static_cast<float>(goal.first);
            float ez = static_cast<float>(goal.second);
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

        // The other explorer, as a 3D soldier, flushing red for a moment when hurt. Solid, so
        // drawn now with depth writes on; the see-through sprites below then sort against it.
        for (int s = 1; s <= 2 && mode != Mode::Single; s++) {
            if (s == my_slot) continue;
            const RemotePlayer& r = remote[s];
            if (!r.present || !r.has_state || !r.alive) continue;
            double feet = floor_height(r.rx, r.ry, r.level) + r.jump;
            glm::vec4 tint = glfwGetTime() < r.flash_until ? glm::vec4(1.8f, 0.4f, 0.4f, 1.0f) : glm::vec4(1.0f);
            draw_player_model(r.rx, feet, r.ry, r.yaw, r.pitch, r.walk_phase, r.walk_amount, eased(r.crouch), tint);
        }

        std::vector<Sprite> sprites;
        for (const auto& m : monsters) {
            if (is_beast() && m.id == beast_boss_id) continue; // you are this one
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

        // The other explorer, if they're on the level being viewed.
        for (int s = 1; s <= 2 && mode != Mode::Single; s++) {
            if (s == my_slot) continue;
            const RemotePlayer& r = remote[s];
            if (!r.present || !r.has_state || !r.alive || r.level != player_level) continue;
            draw_circle(static_cast<float>(r.rx), static_cast<float>(r.ry), 0.34f, player_color(s));
        }

        // Everything below lives on the maze level only.
        if (player_level != 0) {
            draw_circle(static_cast<float>(player_pos_x), static_cast<float>(player_pos_y), 0.4f, { 0.0f, 1.0f, 0.0f, 1.0f });
            draw_line(static_cast<float>(player_pos_x), static_cast<float>(player_pos_y),
                static_cast<float>(player_pos_x + dir_x * 0.8), static_cast<float>(player_pos_y + dir_y * 0.8),
                { 0.0f, 0.5f, 0.0f, 1.0f });
            return;
        }

        if (!is_beast()) {
            auto goal = exit_cell(my_slot);
            draw_quad(static_cast<float>(goal.first), static_cast<float>(goal.second), static_cast<float>(goal.first + 1), static_cast<float>(goal.second + 1), { 1.0f, 0.24f, 0.24f, 1.0f });
        }

        for (const auto& m : monsters) {
            glm::vec4 color = m.type == 1 ? glm::vec4(0.14f, 0.22f, 0.77f, 1.0f) : glm::vec4(0.71f, 0.12f, 0.71f, 1.0f);
            if (beast_active() && m.id == beast_boss_id) color = player_color(3); // the Beast stands out
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
        // The Beast's health is its boss's.
        int hp_now = player_hp, hp_max = max_hp;
        std::string hp_label = "HP: ";
        if (is_beast()) {
            const Monster* m = find_monster(beast_boss_id);
            hp_now = m ? m->hp : 0;
            hp_max = BEAST_MAX_HP;
            hp_label = "Beast HP: ";
        }
        draw_quad(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, { 1.0f, 0.0f, 0.0f, 1.0f });
        float green_w = bar_w * std::clamp(hp_now / static_cast<float>(hp_max), 0.0f, 1.0f);
        draw_quad(bar_x, bar_y, bar_x + green_w, bar_y + bar_h, { 0.0f, 1.0f, 0.0f, 1.0f });

        // Draw text with white color
        draw_text(hp_label + std::to_string(hp_now) + "/" + std::to_string(hp_max), bar_x + bar_w + 10, bar_y + 5, { 1.0f, 1.0f, 1.0f, 1.0f });
        draw_text("Bosses: " + std::to_string(boss_count), bar_x, bar_y - 30, { 1.0f, 1.0f, 1.0f, 1.0f });
        if (player_level != 0)
            draw_text("Lower Level", bar_x, bar_y - 60, { 0.55f, 0.75f, 1.0f, 1.0f });

        if (dev_mode && mode == Mode::Single) draw_text("DEV MODE - F6: boss room  F7: near exit", 10.0f, 30.0f, { 0.75f, 0.75f, 0.75f, 1.0f });

        // Multiplayer: who you are and what you're after, centred along the top, with the kill
        // feed down the right. Both stay clear of the minimap in the top-left corner.
        if (mode != Mode::Single && in_round) {
            float w = static_cast<float>(screen_width);
            float minimap_right = 16.0f + std::max(64.0f, std::min(w, static_cast<float>(screen_height)) / 5.0f);
            std::string goal = my_slot == 1 ? (mode == Mode::Host ? "Player 1 (host): kill the bosses, then reach the exit"
                                                                  : "Player 1: kill the bosses, then reach the exit")
                             : my_slot == 2 ? "Player 2: kill the bosses, then reach Player 1's starting point"
                             : "Player 3 - the Beast: hunt the explorers down";
            float room = w - 2.0f * (minimap_right + 12.0f);
            float s = std::clamp(room / std::max(1.0f, text_width(goal, 1.0f)), 0.4f, 0.75f);
            draw_text_centered(goal, w * 0.5f, 34.0f, player_color(my_slot), s);
            float y = 64.0f;
            for (const auto& f : feed) {
                float fs = std::clamp((w * 0.45f) / std::max(1.0f, text_width(f.first, 1.0f)), 0.4f, 0.62f);
                draw_text(f.first, w - text_width(f.first, fs) - 14.0f, y, { 0.92f, 0.92f, 0.95f, 1.0f }, fs);
                y += 26.0f;
            }
        }

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

    void draw_text(const std::string& text, float x, float y, glm::vec4 color, float scale = 1.0f) {
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

static LaunchOptions parse_args(int argc, char** argv) {
    LaunchOptions o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto value = [&](const std::string& key) -> std::string {
            // Accepts both "--key=value" and "--key value".
            if (a.rfind(key + "=", 0) == 0) return a.substr(key.size() + 1);
            if (a == key && i + 1 < argc) return argv[++i];
            return "";
        };
        if (a == "--windowed") o.windowed = true;
        else if (a == "--windowed=left") { o.windowed = true; o.window_side = 1; }
        else if (a == "--windowed=right") { o.windowed = true; o.window_side = 2; }
        else if (a == "--host") o.host = true;
        else if (a.rfind("--join", 0) == 0) o.join = value("--join");
        else if (a.rfind("--autostart", 0) == 0) o.autostart = std::atoi(value("--autostart").c_str());
        else if (a.rfind("--quit-after", 0) == 0) o.quit_after = std::atof(value("--quit-after").c_str());
        else if (a == "--test") o.test = true;
        else if (a.rfind("--test-size", 0) == 0) {
            std::string v = value("--test-size");
            size_t x = v.find('x');
            if (x != std::string::npos) {
                o.test_w = std::clamp(std::atoi(v.c_str()), 320, 3840);
                o.test_h = std::clamp(std::atoi(v.c_str() + x + 1), 200, 2160);
            }
        }
        else if (a == "--test-fire") o.test_fire = true;
        else if (a.rfind("--test-spawn-near", 0) == 0) {
            o.test_spawn_near = true;
            if (a.size() > 18 && a[17] == '=') o.test_spawn_yaw = std::atof(a.c_str() + 18);
        }
        else if (a == "--open-menu") o.open_menu = true;
        else if (a == "--open-menu=sound") { o.open_menu = true; o.open_sound = true; }
        else if (a.rfind("--screenshot", 0) == 0) o.screenshot = value("--screenshot");
        else if (a.rfind("--test-chat", 0) == 0) o.test_chat = value("--test-chat");
        else std::cerr << "Unknown option: " << a << std::endl;
    }
    return o;
}

int main(int argc, char** argv) {
    try {
        MazeGame game(parse_args(argc, argv));
    }
    catch (const std::exception& e) {
        std::cerr << "Error launching game: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
