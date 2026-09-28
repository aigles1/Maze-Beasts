# MazeBeasts dedicated server

A headless server for MazeBeasts multiplayer (v0.41: games and server must be the same version). Two or three players connect to it directly by IP address and port. It runs the maze's monsters and bosses, decides who gets each medpack and who escaped first, starts the rounds, and relays every player's moves and shots to the others. It uses ENet (reliable UDP) on **UDP port 29180**.

It builds the same maze as the players from a shared seed. The maze generator doesn't depend on the compiler's standard library, so a Linux server built with GCC agrees exactly with the Windows game.

## How a game on the server works

- The **first** player to join is **Player 1**: they start where singleplayer does, and get a **Start the game** button once a second player is in. The second player is **Player 2** (starts at the exit), and the optional third is **Player 3, the Beast**.
- After someone escapes, the next maze starts by itself 5 seconds later. Player 1 can press **F8** for a new maze at any time.
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

Then, from the source bundle (`mazebeasts-server-0.41-src.tar.gz`):

```
tar xzf mazebeasts-server-0.41-src.tar.gz
cd mazebeasts-server-0.41
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

The games and the server must be the same version; a game that doesn't match is told so when it tries to join. To update, copy the new bundle to the machine as before, then:

```
kill $(pidof mazebeasts-server)   # if the old one is running
tar xzf mazebeasts-server-0.41-src.tar.gz
cd mazebeasts-server-0.41
make
./mazebeasts-server
```

The compiler is already installed, so there's no `dnf`/`apt` step this time.

## Running it on AWS EC2

1. **Launch an instance.** In the EC2 console, pick the region nearest to you and the players. Then click **Launch instance**:
   - **AMI:** Amazon Linux 2023, 64-bit (x86).
   - **Instance type:** `t3.micro` (or `t2.micro` if that's the free-tier choice in your region).
   - **Key pair:** create one (RSA, `.pem`) and save the file, for example as `C:\Users\<you>\Downloads\mazebeasts.pem`.
   - **Network settings → Edit → Create security group** with two inbound rules:
     - `SSH`, TCP 22, source **My IP**
     - `Custom UDP`, port **29180**, source **Anywhere-IPv4** (`0.0.0.0/0`), or only the players' IP addresses
   - Keep the default storage, then **Launch instance**. When it shows *Running*, copy its **Public IPv4 address**.
2. **Copy the server source up** from PowerShell on your PC (Windows 10 and 11 include `scp` and `ssh`):
   ```
   scp -i C:\Users\<you>\Downloads\mazebeasts.pem mazebeasts-server-0.41-src.tar.gz ec2-user@<public-ip>:~
   ```
   Answer `yes` the first time it asks about the host's fingerprint. If it refuses with *UNPROTECTED PRIVATE KEY FILE*, restrict the key file to your own account, then try again:
   ```
   icacls C:\Users\<you>\Downloads\mazebeasts.pem /inheritance:r /grant:r "$($env:USERNAME):(R)"
   ```
3. **Build and start it:**
   ```
   ssh -i C:\Users\<you>\Downloads\mazebeasts.pem ec2-user@<public-ip>
   sudo dnf install -y gcc-c++ make
   tar xzf mazebeasts-server-0.41-src.tar.gz
   cd mazebeasts-server-0.41
   make
   ./mazebeasts-server
   ```
4. **Play:** in each game, choose **Esc → Multiplayer - Join**, type the instance's public IP address, and click **Connect**. When everyone is in, Player 1 clicks **Start the game**.
5. **When you're done**, stop the server with Ctrl+C. Then in the EC2 console choose **Instance state → Stop** (you can start it again later) or **Terminate** (deletes it). A stopped instance gets a new public IP address when started again, unless you attach an Elastic IP.

If a game says *Could not reach ...*, check that the server is running, that the IP address is the instance's **public** one, and that the security group has the UDP 29180 rule.

## Windows

The release also includes `MazeBeastsServer.exe`, the same server for Windows (for example, on a LAN). Run it from a console, or double-click it. `Test dedicated server on this PC.bat` starts it together with two game windows already joined to it. The first time, Windows Firewall may ask whether to allow it on your network.
