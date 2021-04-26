CXXFLAGS += -Wall -Wextra -g -O3 -DNDEBUG

.phony: all wsp release

all: release

release: simulator.cpp
	g++ -std=c++11 simulator.cpp -o simulator $(CXXFLAGS)

clean:
	rm -f ./simulator
