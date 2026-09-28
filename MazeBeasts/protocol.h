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
    MSG_STATE,         // any   u8 round, u8 slot, u8 level, u8 flags, f32 x y jump yaw pitch, i16 hp
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
    MSG_CHAT,          // C->S  text (u8 length + bytes); S->C  u8 sender's slot, then the text.
                       //       Not tied to a round: it works in the lobby and between mazes too.
};

// 2 (v0.4): dedicated servers, and a maze generator that no longer depends on the compiler's
// standard library - so a v0.3 copy would build a different maze from the same seed.
// 3 (v0.41): chat. An older server or host would silently drop it, so they don't mix.
constexpr uint8_t PROTOCOL_VERSION = 3;

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
