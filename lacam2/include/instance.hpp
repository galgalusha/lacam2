/*
 * instance definition
 */
#pragma once
#include <memory>
#include <random>

#include "graph.hpp"
#include "utils.hpp"

struct Instance {
private:
  // Owns the graph when constructed from file/grid; null when borrowing from a parent.
  // Must be declared before G so it is constructed first.
  std::unique_ptr<Graph> G_owned;

public:
  const Graph& G;   // always valid; refers to *G_owned or to a parent's G
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
  // sub-problem constructor: borrows parent's graph (no copy).
  // new_starts / new_goals must be vertex pointers from parent.G.
  Instance(const Instance& parent, const Config& new_starts, const Config& new_goals);
  ~Instance() = default;

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
