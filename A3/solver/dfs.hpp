// src/solver/dfs.hpp

#ifndef DFS_HPP
#define DFS_HPP

#include "mazesolver.hpp"

class DFS : public MazeSolver {
public:
    Path solveMaze(const Maze& maze) override;
};

#endif
