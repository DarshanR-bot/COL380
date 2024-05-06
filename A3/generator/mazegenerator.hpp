#ifndef MAZEGENERATOR_HPP
#define MAZEGENERATOR_HPP

#include <vector>

struct Cell {
    int x, y;
    bool walls[4]; // 0: up, 1: right, 2: down, 3: left
    Cell(int x = 0, int y = 0) : x(x), y(y) {
        walls[0] = walls[1] = walls[2] = walls[3] = true;
    }
};

using Maze = std::vector<std::vector<Cell>>;

class MazeGenerator {
public:
    virtual Maze generateMaze(int width, int height) = 0;
    virtual ~MazeGenerator() {}  // Virtual destructor
};

#endif
