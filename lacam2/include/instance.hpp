/*
 * instance definition
 */
#pragma once
#include <random>

#include "graph.hpp"
#include "utils.hpp"

struct Instance {
  const Graph G;
  Config starts;
  Config goals;
  const uint N;  // number of agents

  // for testing
  Instance(const std::string& map_filename,
           const std::vector<uint>& start_indexes,
           const std::vector<uint>& goal_indexes);
  // for testing with in-memory grid ('.' = passable)
  Instance(const std::vector<std::string>& grid,
           const std::vector<uint>& start_indexes,
           const std::vector<uint>& goal_indexes);
  // for MAPF benchmark
  Instance(const std::string& scen_filename, const std::string& map_filename,
           const uint _N = 1);
  // random instance generation
  Instance(const std::string& map_filename, std::mt19937* MT,
           const uint _N = 1);
  ~Instance() {}

  // create a new instance by tiling this grid n times horizontally
  // with one wall column between adjacent copies
  Instance multiply(int n) const;

  // print ASCII map: '.' free, '#' wall, and starts as agent_id % 10
  void render(std::ostream& os = std::cout) const;

  // simple feasibility check of instance
  bool is_valid(const int verbose = 0) const;
};

// solution: a sequence of configurations
using Solution = std::vector<Config>;
std::ostream& operator<<(std::ostream& os, const Solution& solution);
