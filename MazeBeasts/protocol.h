// The multiplayer protocol, shared by the game and the dedicated server.
//
// One machine is the authority - a player hosting from the menu, or a dedicated server - and
// everyone else connects to it directly by IP. The authority runs the monsters, bosses,
// medpacks and rounds, and relays every player's messages to the rest. Each player is the
// authority on their own movement and health: the shooter's machine decides damage to
// monsters, and a victim's machine decides damage to itself.
//
// Slots: 1 = starts where singleplayer does (the host, when a player hosts); 2 = starts at the
// exit and must escape through Player 1's start; 3 = optional, controls one of the bosses
// ("the Beast").
//
// Every message starts with its type byte. Gameplay messages then carry the round number, so
// anything still in flight from a previous maze is recognised and ignored.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

enum Msg : uint8_t {
    MSG_HELLO = 1,     // C->S  u8 protocol version
    MSG_WELCOME,       // S->C  u8 slot, u32 seed, u8 flags (WelcomeFlags)
    MSG_REJECT,        // S->C  u8 reason
    MSG_LOBBY,         // S->C  u8 bit mask of occupied slots
    MSG_ASSIGN,        // S->C  u8 new slot (players move up to fill a gap in the lobby)
    MSG_START,         // S->C  u8 round, u32 seed, i32 Beast's boss id, u8 slot mask
    MSG_STATE,         // any   u8 round, u8 slot, u8 level, u8 flags (StateFlags), f32 x y jump yaw pitch, i16 hp
    MSG_SHOT,          // any   u8 round, u8 owner, u8 level, u8 boss shot, f32 x y z dx dy dz speed
    MSG_MONSTERS,      // S->C  u8 round, u16 count, count x {i32 id, f32 x y, i32 hp, u8 type, u8 flash}
    MSG_MONSTER_HIT,   // C->S  u8 round, i32 monster id, i32 damage
    MSG_DEATH,         // any   u8 round, u8 victim, u8 killer (0 = a monster)
    MSG_PICKUP,        // C->S  u8 round, i32 medpack id
    MSG_PACK_GONE,     // S->C  u8 round, i32 medpack id
    MSG_REACHED_EXIT,  // C->S  u8 round, u8 slot
    MSG_ROUND_OVER,    // S->C  u8 round, u8 winner
    MSG_BEAST_BOSS,    // S->C  u8 round, i32 boss id the Beast now controls (-1 = none left)
    MSG_REQUEST_START, // C->S  dedicated server only: Player 1 asks for the first maze, or a new one
    MSG_TO_LOBBY,      // S->C  dedicated server only: no explorers left, so the round is abandoned
    MSG_CHAT,          // C->S  text (u8 length + bytes); S->C  u8 sender's slot (0 = the game
                       //       itself, e.g. about votes), then the text. Not tied to a round:
                       //       it works in the lobby and between mazes too.
    MSG_VOTE_START,    // C->S  (F8, or "votemap" in chat) start a vote for a new maze
    MSG_VOTE_CAST,     // C->S  u8 1 = Yes, 0 = No
    MSG_VOTE,          // S->C  u8 vote id, u8 starter, u8 voters mask, u8 yes mask, u8 no mask,
                       //       u8 VoteState, u16 milliseconds left
};

// 2 (v0.4): dedicated servers, and a maze generator that no longer depends on the compiler's
// standard library - so a v0.3 copy would build a different maze from the same seed.
// 3 (v0.41): chat. An older server or host would silently drop it, so they don't mix.
// 4 (v0.42): crouching, and voting for a new maze.
constexpr uint8_t PROTOCOL_VERSION = 4;

enum StateFlags : uint8_t {
    STATE_ALIVE     = 1,
    STATE_CROUCHING = 2,
};

enum RejectReason : uint8_t { REJECT_IN_PROGRESS = 1, REJECT_VERSION = 2, REJECT_FULL = 3 };

enum WelcomeFlags : uint8_t {
    WELCOME_DEDICATED   = 1, // a dedicated server: Player 1 is a client too, and starts the rounds
    WELCOME_IN_PROGRESS = 2, // a round is under way; you join in at the next maze
};

constexpr double STATE_INTERVAL     = 1.0 / 30.0; // each player's own position, 30 times a second
constexpr double MONSTER_INTERVAL   = 1.0 / 20.0; // the authority's monster snapshot
constexpr double ROUND_OVER_SECONDS = 5.0;        // winner banner before the next maze

constexpr size_t CHAT_MAX_CHARS = 200;

// What a chat message may contain: the game's font has printable ASCII only, so anything else
// becomes '?' (one per character), line breaks and tabs become spaces, and it is cut at
// CHAT_MAX_CHARS. Senders, the server and receivers all apply it, so nobody has to trust
// what arrives over the network.
inline std::string clean_chat_text(const std::string& in, size_t max_chars = CHAT_MAX_CHARS) {
    std::string out;
    for (size_t i = 0; i < in.size() && out.size() < max_chars; i++) {
        unsigned char c = static_cast<unsigned char>(in[i]);
        if (c == '\n' || c == '\r' || c == '\t') {
            if (!out.empty() && out.back() != ' ') out.push_back(' ');
        }
        else if (c >= 32 && c < 127) out.push_back(static_cast<char>(c));
        else if (c >= 0xC0) out.push_back('?'); // the first byte of a UTF-8 character; the rest are skipped
    }
    return out;
}

inline std::string trim_spaces(const std::string& s) {
    size_t a = s.find_first_not_of(' ');
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(' ') - a + 1);
}

// ---- Voting for a new maze ----------------------------------------------------------------
// The rules, kept by whoever runs the game (a hosting player, or the dedicated server) so they
// are the same everywhere:
//  - Starting a vote counts as a Yes.
//  - It passes with 2 Yes votes (or every player's, when fewer than 2 are playing), and then a
//    new maze starts, just as F8 does in singleplayer.
//  - It fails as soon as 2 Yes votes are out of reach (say, 2 players vote No), and it expires
//    after VOTE_SECONDS.
//  - A player who starts VOTE_SOLO_LIMIT votes in a row that nobody else votes in must wait
//    VOTE_COOLDOWN_SECONDS before starting another. A vote that others take part in (and any
//    that passes) clears that count, so there's no limit on votes people actually want.

constexpr double VOTE_SECONDS          = 17.0;
constexpr int    VOTE_SOLO_LIMIT       = 5;
constexpr double VOTE_COOLDOWN_SECONDS = 240.0;

enum VoteState : uint8_t { VOTE_OPEN = 0, VOTE_PASSED = 1, VOTE_FAILED = 2, VOTE_EXPIRED = 3 };

class MapVote {
public:
    bool active = false;
    uint8_t id = 0;          // counts up with each vote
    int starter = 0;
    uint8_t voters = 0, yes = 0, no = 0; // bit per slot
    double deadline = 0.0;

    // Returns 0 when the vote starts. Otherwise: -1 if one is already running, or the number
    // of seconds the player must still wait.
    int start(int slot, uint8_t players, double now) {
        if (slot < 1 || slot > 3) return -1;
        if (active) return -1;
        if (now < cooldown_until[slot]) return static_cast<int>(cooldown_until[slot] - now) + 1;
        active = true;
        id++;
        starter = slot;
        voters = static_cast<uint8_t>(players | bit(slot));
        yes = bit(slot);
        no = 0;
        deadline = now + VOTE_SECONDS;
        return 0;
    }

    // A player's Yes or No; false if they can't vote in this one or already have.
    bool cast(int slot, bool vote_yes) {
        if (slot < 1 || slot > 3 || !active) return false;
        uint8_t b = bit(slot);
        if (!(voters & b) || ((yes | no) & b)) return false;
        if (vote_yes) yes |= b; else no |= b;
        return true;
    }

    // A player left: they no longer count, and their record is forgotten.
    void remove_player(int slot) {
        if (slot < 1 || slot > 3) return;
        uint8_t keep = static_cast<uint8_t>(~bit(slot));
        voters &= keep;
        yes &= keep;
        no &= keep;
        solo[slot] = 0;
        cooldown_until[slot] = 0.0;
    }

    // Settles the vote if it can be settled; VOTE_OPEN while it's still undecided.
    VoteState check(double now) {
        if (!active) return VOTE_OPEN;
        int need = count(voters) < 2 ? count(voters) : 2;
        int pending = count(static_cast<uint8_t>(voters & ~(yes | no)));
        VoteState result = VOTE_OPEN;
        if (need == 0) result = VOTE_FAILED;
        else if (count(yes) >= need) result = VOTE_PASSED;
        else if (count(yes) + pending < need) result = VOTE_FAILED;
        else if (now >= deadline) result = VOTE_EXPIRED;
        if (result != VOTE_OPEN) finish(result, now);
        return result;
    }

    // A new maze started some other way: drop the vote without counting it against anyone.
    void cancel() { active = false; }

    double seconds_left(double now) const { return active && deadline > now ? deadline - now : 0.0; }

private:
    int solo[4] = {};              // votes in a row that only their starter voted in
    double cooldown_until[4] = {};

    static uint8_t bit(int slot) { return static_cast<uint8_t>(1u << slot); }
    static int count(uint8_t m) { int n = 0; for (; m; m = static_cast<uint8_t>(m & (m - 1))) n++; return n; }

    void finish(VoteState result, double now) {
        active = false;
        bool others_voted = ((yes | no) & ~bit(starter)) != 0;
        if (result == VOTE_PASSED || others_voted) solo[starter] = 0;
        else if (++solo[starter] >= VOTE_SOLO_LIMIT) {
            solo[starter] = 0;
            cooldown_until[starter] = now + VOTE_COOLDOWN_SECONDS;
        }
    }
};

// "3:05" - for telling someone how long to wait.
inline std::string minutes_seconds(int seconds) {
    std::string s = std::to_string(seconds % 60);
    return std::to_string(seconds / 60) + ":" + (s.size() < 2 ? "0" + s : s);
}
