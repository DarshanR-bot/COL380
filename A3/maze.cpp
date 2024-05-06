#include <mpi.h>
#include <iostream>
#include <cstring>
#include "generator/mazegenerator.hpp"
#include "solver/mazesolver.hpp"
#include "generator/bfs.hpp"
#include "generator/kruskal.hpp"
#include "solver/dfs.hpp"
#include "solver/dijkstra.hpp"

bool generate_with_bfs = false;
bool solve_with_dfs = false;

void printMaze(const Maze& maze, const Path& path);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "bfs") == 0) {
                generate_with_bfs = true;
            }
        }
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "dfs") == 0) {
                solve_with_dfs = true;
            }
        }
    }

    MazeGenerator* generator;
    MazeSolver* solver;

    // Instantiate appropriate generator and solver based on command-line arguments
    if (generate_with_bfs) {
        generator = new BFS();
    } else {
        generator = new Kruskal();
    }
    
    if (solve_with_dfs) {
        solver = new DFS();
    } else {
        solver = new Dijkstra();
    }
    
    // Generate and solve the maze
    Maze maze = generator->generateMaze(64, 64);
    Path path = solver->solveMaze(maze);
    
    // Output result
    if (world_rank == 0) {  // Assuming output should come from a single process
        printMaze(maze, path);
    }
    
    delete generator;
    delete solver;
    
    MPI_Finalize();
    return 0;
}
