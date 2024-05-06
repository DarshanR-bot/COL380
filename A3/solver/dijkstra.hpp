// src/solver/dijkstra.hpp

#ifndef DIJKSTRA_HPP
#define DIJKSTRA_HPP

#include "mazesolver.hpp"

class Dijkstra : public MazeSolver {
public:
    Path solveMaze(const Maze& maze) override;
};

#endif
