#include "bfs.hpp"
#include <queue>
#include <cstdlib>
#include <vector>
#include <utility>
#include <algorithm>

// Function to generate the maze
Maze BFS::generateMaze(int width, int height) {
    Maze maze(width, height);
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));
    std::queue<std::pair<int, int>> queue;

    // Random starting point
    int startX = rand() % width;
    int startY = rand() % height;
    queue.push({startX, startY});
    visited[startY][startX] = true;

    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop();
        exploreNeighbors(maze, current.first, current.second, visited);
    }

    return maze;
}

// Helper function to explore neighbors
void BFS::exploreNeighbors(Maze &maze, int x, int y, std::vector<std::vector<bool>> &visited) {
    std::vector<std::pair<int, int>> directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    std::random_shuffle(directions.begin(), directions.end());  // Randomize directions to ensure random maze generation

    for (auto &dir : directions) {
        int nx = x + dir.first;
        int ny = y + dir.second;

        if (nx >= 0 && nx < maze.getWidth() && ny >= 0 && ny < maze.getHeight() && !visited[ny][nx]) {
            maze.removeWall(x, y, nx, ny);  // This function should remove the wall between (x, y) and (nx, ny)
            visited[ny][nx] = true;
            queue.push({nx, ny});
        }
    }
}
