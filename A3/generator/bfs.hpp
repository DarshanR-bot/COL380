// src/generator/bfs.hpp

#ifndef BFS_HPP
#define BFS_HPP

#include "mazegenerator.hpp"

class BFS : public MazeGenerator {
public:
    Maze generateMaze(int width, int height) override;
};

#endif
