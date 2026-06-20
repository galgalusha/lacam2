#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

#include <random>

// Abstract base class for neighbor selection policy used in PolicyPIBT.
// Given a config, an agent id, and a list of candidate neighbors with the same
// distance-to-goal value, returns the preferred next vertex.
class Policy {
 public:
  virtual ~Policy() = default;
  virtual Vertex* get_preferred_neighbor(const Config& C, uint agent_id,
                                         const Vertices& candidates) = 0;
};

// Naive policy: random choice among candidates.
class RandomPolicy : public Policy {
 public:
  explicit RandomPolicy(std::mt19937* _MT) : MT(_MT) {}
  Vertex* get_preferred_neighbor(const Config& C, uint agent_id,
                                 const Vertices& candidates) override;

 private:
  std::mt19937* MT;
};
