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
| **Tab** | Show the entire maze |
| **F8** | Generate a new maze (in multiplayer, host only) |
| **Esc** | Open the menu (choose **Exit** there to close the game) |

While spectating in multiplayer, **W A S D** and the mouse fly you around, **Spacebar** goes up and **Ctrl** (or **C**) goes down.

## Sound

**Esc → Sound** has a volume slider: drag it, or use the **Left** and **Right** arrow keys. **Test sound** plays a sample at the new volume. The setting is saved in `%APPDATA%\MazeBeasts\settings.txt`, so it carries over to new releases.

## Multiplayer

Press **Esc** for the menu: **Singleplayer**, **Multiplayer - Join**, **Multiplayer - Host**, **Sound** and **Exit**.

- **Hosting:** choose *Multiplayer - Host*. The lobby shows your IP address and a maze seed, and says *Waiting for players*. Once a second player joins it shows *2/3 players joined* and you can click **Start the game**.
- **Joining:** choose *Multiplayer - Join*, type the host's IP address and press **Connect**. The game uses **UDP port 29180**; to reach a different port, type `address:port`.

**Two or three players:**

- **Player 1** (the host) starts where singleplayer does, and has the same goal: kill the bosses, then reach the exit.
- **Player 2** starts at the exit, and must escape through Player 1's starting point.
- Both players hunt the same bosses, so every boss killed helps both of you toward your own exit.
- The explorers are soldiers in urban camouflage with blue helmets, and they can shoot each other. A killed player spectates for 5 seconds, flying freely around the maze, then respawns at their own starting point.
- **Player 3** is optional, and controls one of the bosses, *the Beast*. If that boss dies, Player 3 takes over another surviving boss. With no boss left, Player 3 spectates until the next maze.

**Playing across computers:**

- The host may see a Windows Firewall prompt the first time they host. Allow it on private networks for LAN play.
- Over the internet, the host must forward UDP port 29180 on their router.

**Trying it on one PC:** run `Test multiplayer on this PC.bat` from the release folder. It opens two windowed copies side by side, and you connect one to the other with `127.0.0.1`. You can also start copies yourself with `MazeBeasts.exe --windowed=left` and `--windowed=right`.

## Why C++

This is better than my original [Python version](https://github.com/aigles1/pyMaze-Beasts), since this one is written in C++. With C++ the game performs better and is more comfortable on the eyes. In particular, the walls don't warp anymore.

## Download and play

Download the latest release zip from the [Releases](../../releases) page and unzip it. Keep `MazeBeasts.exe` and `assets.dat` in the same folder, then run `MazeBeasts.exe`.

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

## Future plans

I might get rid of the projectiles in the future and just show damage on the walls and monsters.
Update: 9/28 I have a working multiplayer version, will upload soon

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
