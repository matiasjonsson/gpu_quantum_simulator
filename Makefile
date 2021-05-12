CXXFLAGS += -Wall -Wextra -g -O3 -DNDEBUG

.phony: all wsp release

all: release


NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc
LIBS += GL glut cudart


LDLIBS  := $(addprefix -l , $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc


release: simulator.cpp
	g++ -std=c++11 simulator.cpp -o simulator $(CXXFLAGS)
	nvcc gpu_simulator.cu -o fastsimulator $(NVCCFLAGS) $(LDLIBS)
clean:
	rm -f ./simulator
