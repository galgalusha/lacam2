#pragma once
#include <instance.hpp>

// Launch the SFML + Dear ImGui visualizer.
// Draws the map, obstacles, agents (colored circles) and goals (colored
// rectangles) for the given instance.  Blocks until the window is closed.
void run_gui(const Instance& ins);
