# Builds the MazeBeasts dedicated server on Linux: run `make`, then ./mazebeasts-server
# Needs only a C/C++ compiler (g++ with C++17) and make.
#
# Works both in the repository (shared sources in ../MazeBeasts) and in the release's server
# source bundle (everything in one folder).

SHARED := $(if $(wildcard world.cpp),.,../MazeBeasts)
ENET   := $(SHARED)/enet
SERVER := $(if $(wildcard server.cpp),.,../MazeBeastsServer)

CC  ?= gcc
CXX ?= g++

# What ENet's own build would detect on a modern Linux.
ENET_DEFS := -DHAS_POLL=1 -DHAS_FCNTL=1 -DHAS_INET_PTON=1 -DHAS_INET_NTOP=1 -DHAS_MSGHDR_FLAGS=1 \
             -DHAS_SOCKLEN_T=1 -DHAS_GETADDRINFO=1 -DHAS_GETNAMEINFO=1

CPPFLAGS += -I$(SHARED) -I$(ENET)/include
# -ffp-contract=off: no fused multiply-adds, so the maze generator's arithmetic comes out
# exactly as it does in the Windows game (it matters on ARM64, e.g. Graviton instances).
CFLAGS   += -O2 -ffp-contract=off
CXXFLAGS += -std=c++17 -O2 -Wall -ffp-contract=off

ENET_SRCS := callbacks.c compress.c host.c list.c packet.c peer.c protocol.c unix.c
ENET_OBJS := $(ENET_SRCS:%.c=enet_%.o)
OBJS      := server.o world.o net.o $(ENET_OBJS)
HEADERS   := $(SHARED)/world.h $(SHARED)/net.h $(SHARED)/protocol.h

mazebeasts-server: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS)

server.o: $(SERVER)/server.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

world.o: $(SHARED)/world.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

net.o: $(SHARED)/net.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

enet_%.o: $(ENET)/%.c
	$(CC) $(CPPFLAGS) $(ENET_DEFS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) mazebeasts-server

.PHONY: clean
