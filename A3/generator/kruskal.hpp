// src/generator/kruskal.hpp

#ifndef KRUSKAL_HPP
#define KRUSKAL_HPP

#include "mazegenerator.hpp"

class Kruskal : public MazeGenerator {
public:
    Maze generateMaze(int width, int height) override;
};

#endif
