# Maze-Beasts

A first-person procedural maze shooter written in C++ with OpenGL.

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
| **F8** | Generate a new maze |
| **Esc** | Close the game |

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

I might get rid of the projectiles in the future and just show damage on the walls and monsters. I'll update this when I have more ideas.

## License

Maze-Beasts is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).

It includes third-party components under their own licenses:

- [Liberation Sans](https://github.com/liberationfonts/liberation-fonts) font: SIL Open Font License 1.1 (see `MazeBeasts/LiberationSans-LICENSE.txt`)
- [GLFW](https://www.glfw.org/): zlib/libpng License
- [GLM](https://github.com/g-truc/glm): MIT License
- [glad](https://github.com/Dav1dde/glad): generated OpenGL loader
- [miniaudio](https://miniaud.io/): public domain / MIT No Attribution
- [stb_image and stb_truetype](https://github.com/nothings/stb): public domain / MIT
