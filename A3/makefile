CXX = mpic++
CXXFLAGS = -std=c++11 -Wall -g

all: maze.out

maze.out: maze.o
    $(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
    $(CXX) $(CXXFLAGS) -c $<

clean:
    rm -f *.o *.out
