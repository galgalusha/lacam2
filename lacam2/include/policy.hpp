#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

#include <random>
#include <unordered_map>

// Abstract base class for neighbor selection policy used in PolicyPIBT.
// Given a config, an agent id, and all neighbor vertices (including self),
// returns a score in [-0.9, 0] for each neighbor. -0.9 = most preferred.
// PolicyPIBT adds these scores as tie-breakers to D values.
class Policy {
 public:
  virtual ~Policy() = default;
  virtual std::unordered_map<Vertex*, float> get_neighbor_scores(
      const Config& C, uint agent_id, const Vertices& neighbors) = 0;
};

// Naive policy: all zeros (no preference, equivalent to vanilla PIBT random tie-breaking).
class RandomPolicy : public Policy {
 public:
  explicit RandomPolicy(std::mt19937* _MT) : MT(_MT) {}
  std::unordered_map<Vertex*, float> get_neighbor_scores(
      const Config& C, uint agent_id, const Vertices& neighbors) override;

 private:
  std::mt19937* MT;
};

struct NeighborScores {
  std::unordered_map<Vertex*, int> scores;  // neighbor vertex -> score
};

struct AgentPolicy {
  std::unordered_map<Vertex*, NeighborScores> vertex_scores;  // from-vertex -> neighbor scores

  void record_move(Vertex* from, Vertex* to);
};

// Policy backed by learned neighbor scores.
class NeighborScorePolicy : public Policy {
 public:
  explicit NeighborScorePolicy(std::vector<AgentPolicy> agent_policies,
                               std::mt19937* _MT)
      : policies(std::move(agent_policies)), MT(_MT) {}

  std::unordered_map<Vertex*, float> get_neighbor_scores(
      const Config& C, uint agent_id, const Vertices& neighbors) override;

  uint random_count = 0;
  uint deterministic_count = 0;

 private:
  std::vector<AgentPolicy> policies;
  std::mt19937* MT;
};
