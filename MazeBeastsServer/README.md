# MazeBeasts dedicated server

A headless server for MazeBeasts multiplayer (v0.42: games and server must be the same version). Two or three players connect to it directly by IP address and port. It runs the maze's monsters and bosses, decides who gets each medpack and who escaped first, starts the rounds, and relays every player's moves and shots to the others. It uses ENet (reliable UDP) on **UDP port 29180**.

It builds the same maze as the players from a shared seed. The maze generator doesn't depend on the compiler's standard library, so a Linux server built with GCC agrees exactly with the Windows game.

## How a game on the server works

- The **first** player to join is **Player 1**: they start where singleplayer does, and get a **Start the game** button once a second player is in. The second player is **Player 2** (starts at the exit), and the optional third is **Player 3, the Beast**.
- After someone escapes, the next maze starts by itself 5 seconds later. During a maze, anyone can start a vote for a new one (**F8**, or `votemap` in chat); the server counts the votes and starts the new maze when 2 players say Yes. The rules are in the game's README.
- Someone who joins while a maze is under way waits in the lobby and plays from the next maze.
- If a player leaves the lobby, the others move up (Player 2 becomes Player 1, and so on).
- If every explorer leaves mid-maze, that maze ends and anyone left goes back to the lobby.
- Chat is passed on to the other players, and each message also appears in the server's log.

## Build and run on Linux

You need `g++` (C++17) and `make`:

```
sudo dnf install -y gcc-c++ make        # Amazon Linux 2023
sudo apt install -y g++ make            # Ubuntu / Debian
```

Then, from the source bundle (`mazebeasts-server-0.42-src.tar.gz`):

```
tar xzf mazebeasts-server-0.42-src.tar.gz
cd mazebeasts-server-0.42
make
./mazebeasts-server
```

It prints `listening on UDP port 29180` and logs players joining, mazes starting (with a checksum of the maze) and who escaped. Stop it with **Ctrl+C**.

Options:

| Option | Meaning |
|---|---|
| `--port N` | Listen on UDP port N instead of 29180. Players then join with `address:N`. |
| `--autostart N` | Start the first maze by itself once N players (2 or 3) have joined, instead of waiting for Player 1's Start button. |
| `--seed N` | Use seed N for the first maze. Later mazes are random. |

To keep it running after you log out:

```
nohup ./mazebeasts-server > server.log 2>&1 &
tail -f server.log                # watch it (Ctrl+C stops watching, not the server)
pidof mazebeasts-server           # is it running? prints its process number, or nothing
kill $(pidof mazebeasts-server)   # stop it
```

(`pkill mazebeasts-server` doesn't work: Linux shortens process names to 15 characters.)

From the repository (instead of the bundle), run `make` in this `MazeBeastsServer` folder.

### Updating to a new version

The games and the server must be the same version; a game that doesn't match is told so when it tries to join. To update, copy the new bundle to the server machine, then:

```
kill $(pidof mazebeasts-server)   # if the old one is running
tar xzf mazebeasts-server-0.42-src.tar.gz
cd mazebeasts-server-0.42
make
./mazebeasts-server
```

The compiler is already installed, so there's no `dnf`/`apt` step this time.

If a game says *Could not reach ...*, check that the server is running, that you typed the server's public IP address, and that UDP port 29180 is allowed through any firewall in between.

## Windows

The release also includes `MazeBeastsServer.exe`, the same server for Windows (for example, on a LAN). Run it from a console, or double-click it. To try it on one PC, start two games that join it: `MazeBeasts.exe --windowed=left --join=127.0.0.1` and `MazeBeasts.exe --windowed=right --join=127.0.0.1`. The first time, Windows Firewall may ask whether to allow it on your network.
