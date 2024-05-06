#ifndef MAZESOLVER_HPP
#define MAZESOLVER_HPP

#include "../generator/mazegenerator.hpp"
#include <list>
typedef std::list<Cell> Path;

class MazeSolver {
public:
    virtual Path solveMaze(const Maze& maze) = 0;
    virtual ~MazeSolver() {}  // Virtual destructor
};

#endif
