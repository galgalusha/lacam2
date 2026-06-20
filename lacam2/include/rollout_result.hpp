#pragma once

#include "graph.hpp"

#include <vector>

struct RolloutResult {
  bool success;
  uint cost;
  uint makespan;
  std::vector<Config> configs;  // sequence of configs produced during rollout
};
