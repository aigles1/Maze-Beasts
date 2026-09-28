# Maze-Beasts

A first-person procedurally generated maze shooter written in C++ with OpenGL.

## Objective

Defeat the bosses and find the exit of the maze.

## Controls

| Key | Action |
|---|---|
| **W A S D** | Move |
| **Mouse** | Look and aim |
| **Left click** | Shoot |
| **Spacebar** | Jump |
| **Ctrl** (hold) | Crouch: lower and slower, and a smaller target |
| **Tab** | Show the entire maze |
| **F8** | Generate a new maze (in multiplayer: start a vote for one) |
| **F1 / F2** | Vote Yes / No on a new maze (multiplayer) |
| **F5** | Developer mode (singleplayer only; then F6 and F7 teleport) |
| **Enter** | Chat: type a message (up to 200 characters), then **Enter** again to send it |
| **Y** | Open or close the chat log |
| **Ctrl+C / Ctrl+V** | Copy and paste, in chat and in the Join screen's address box |
| **Esc** | Close the chat line or chat log; otherwise open the menu (choose **Exit** there to close the game) |

While spectating in multiplayer, **W A S D** and the mouse fly you around, **Spacebar** goes up and **Ctrl** (or **C**) goes down.

**Esc → Controls** shows these in the game.

## Chat

Press **Enter**, type, and press **Enter** again to send (**Esc** cancels). Messages appear along the bottom centre of everyone's screen for 15 seconds, with up to 8 showing at once: a new one pushes the oldest off the top.

**Y** opens the chat log, a small window with everything said since the game started. **Up** and **Down** pick a message and **Ctrl+C** copies it; the mouse wheel, **Page Up**, **Page Down**, **Home** and **End** scroll. The log is kept in memory only, so it's gone when you close the game.

**Ctrl+V** pastes into the chat line, and into the Join screen's address box (replacing what's there). **Ctrl+C** copies what you've typed, or on the Host screen copies your IP address to send to the others. The game's font shows plain English letters, numbers and punctuation; other characters appear as `?`.

## Sound

**Esc → Sound** has a volume slider: drag it, or use the **Left** and **Right** arrow keys. **Test sound** plays a sample at the new volume. The setting is saved in `%APPDATA%\MazeBeasts\settings.txt`, so it carries over to new releases.

## Multiplayer

Press **Esc** for the menu: **Singleplayer**, **Multiplayer - Join**, **Multiplayer - Host**, **Sound**, **Controls** and **Exit**.

- **Hosting:** choose *Multiplayer - Host*. The lobby shows your IP address and a maze seed, and says *Waiting for players*. Once a second player joins it shows *2/3 players joined* and you can click **Start the game**.
- **Joining:** choose *Multiplayer - Join*, type the host's (or dedicated server's) IP address and press **Connect**. The game uses **UDP port 29180**; to reach a different port, type `address:port`.

**Two or three players:**

- **Player 1** (the host, or the first to join a dedicated server) starts where singleplayer does, and has the same goal: kill the bosses, then reach the exit.
- **Player 2** starts at the exit, and must escape through Player 1's starting point.
- Both players hunt the same bosses, so every boss killed helps both of you toward your own exit.
- The explorers are soldiers in urban camouflage with blue helmets, and they can shoot each other. A killed player spectates for 5 seconds, flying freely around the maze, then respawns at their own starting point.
- **Player 3** is optional, and controls one of the bosses, *the Beast*. If that boss dies, Player 3 takes over another surviving boss. With no boss left, Player 3 spectates until the next maze.

**Voting for a new maze:** press **F8**, or type `votemap` in chat, to start a vote; starting it counts as your Yes. Everyone else votes with **F1** (Yes) or **F2** (No).

- It passes with 2 Yes votes (or with your own, if you're the only player). A new maze then starts for everyone, like F8 in singleplayer.
- It fails as soon as 2 Yes votes are out of reach, for example when 2 players vote No.
- If it isn't decided within 17 seconds, it expires.
- If you start 5 votes in a row that nobody else answers, you have to wait 4 minutes before starting another. Votes that others take part in, and any that pass, never count against you.

**Playing across computers:**

- The host may see a Windows Firewall prompt the first time they host. Allow it on private networks for LAN play.
- Over the internet, the host must forward UDP port 29180 on their router.

**Trying it on one PC:** from a command prompt in the game's folder, start two copies side by side with `MazeBeasts.exe --windowed=left` and `MazeBeasts.exe --windowed=right`. Host in one, and join `127.0.0.1` from the other.

## Dedicated server

Instead of one player hosting, everyone can join a **dedicated server**: a separate program with no window that runs the maze, its monsters and the rounds. It builds and runs on Linux, so it can live on a cloud machine such as an AWS EC2 instance, and nobody has to forward ports on their router.

- Players join it with *Multiplayer - Join* and its IP address. The first to join is Player 1 and starts the game from the lobby. After that, a new maze takes a vote (**F8** or `votemap`).
- Someone who joins while a maze is under way plays from the next maze.
- The release includes `MazeBeastsServer.exe`, the server for Windows (for a LAN, or to try it on one PC), and `mazebeasts-server-<version>-src.tar.gz`, its source for building on Linux.
- To try it on one PC, run `MazeBeastsServer.exe`, then start two games that join it: `MazeBeasts.exe --windowed=left --join=127.0.0.1` and `MazeBeasts.exe --windowed=right --join=127.0.0.1`. Games and server must be the same version.

See [MazeBeastsServer/README.md](MazeBeastsServer/README.md) to build and run it on Linux.

## Why C++

This is better than my original [Python version](https://github.com/aigles1/pyMaze-Beasts), since this one is written in C++. With C++ the game performs better and is more comfortable on the eyes. In particular, the walls don't warp anymore.

## Download and play

Download the latest release zip from the [Releases](../../releases) page and unzip it. Keep `MazeBeasts.exe` and `assets.dat` in the same folder, then run `MazeBeasts.exe`. (`MazeBeastsServer.exe` is only needed to run a dedicated server.)

Tested on Windows 11. No Visual C++ Redistributable or other installs are needed.

## Building from source

1. Open `MazeBeasts.sln` in Visual Studio 2026 (platform toolset v145).
2. Build the **Release | x64** configuration.
3. Pack the assets:
   ```
   powershell -ExecutionPolicy Bypass -File tools\packassets.ps1
   ```
4. Put `x64\Release\MazeBeasts.exe` and `MazeBeasts\assets.dat` in the same folder and run the exe.

When run from Visual Studio, the game falls back to the loose asset files in the project folder, so step 3 is only needed for a standalone copy.

The solution also builds the dedicated server, `x64\Release\MazeBeastsServer.exe`. For Linux, run `make` in `MazeBeastsServer`, or pack a self-contained source bundle to copy to a Linux machine:

```
powershell -ExecutionPolicy Bypass -File tools\make-server-bundle.ps1
```

## Future plans

I might get rid of the projectiles in the future and just show damage on the walls and monsters.
Eventually I will make a story about UN soldiers training in a facility.

## License

Maze-Beasts is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).

It includes third-party components under their own licenses:

- [Liberation Sans](https://github.com/liberationfonts/liberation-fonts) font: SIL Open Font License 1.1 (see `MazeBeasts/LiberationSans-LICENSE.txt`)
- [GLFW](https://www.glfw.org/): zlib/libpng License
- [GLM](https://github.com/g-truc/glm): MIT License
- [glad](https://github.com/Dav1dde/glad): generated OpenGL loader
- [miniaudio](https://miniaud.io/): public domain / MIT No Attribution
- [stb_image and stb_truetype](https://github.com/nothings/stb): public domain / MIT
- [ENet](https://github.com/lsalzman/enet): MIT License (see `MazeBeasts/enet/LICENSE`)
