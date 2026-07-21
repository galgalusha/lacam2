/*
 * PIBTPlanner: runs 10 Scatter instances with different random seeds,
 * one PIBT per Scatter, 10 rollouts each. Returns the globally best rollout.
 */
#pragma once

#include "planner.hpp"
#include "rollout_result.hpp"
#include "scatter.hpp"
#include "wait_scatter.hpp"
#include <array>
#include <memory>
#include <vector>

// Buckets: 0,1,2,3,4,5,7,10,15,20,50,100 matching steps per agent
struct ScatterBuckets {
  static constexpr std::array<int, 12> THRESHOLDS = {0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 50, 100};
  // counts[i] = number of agents with exactly THRESHOLDS[i] matching steps
  // counts[12] = agents matching more than 100 steps
  std::array<int, 13> counts = {};
  int margin = -1;
};

class PIBTPlanner : public Planner {
 public:
  PIBTPlanner(const Instance* _ins, const Deadline* _deadline,
              std::mt19937* _MT, const int _verbose = 0,
              const Objective _objective = OBJ_NONE,
              const float _restart_rate = 0.001);

  Solution solve(std::string& additional_info);

 private:
  int find_scatter_margin(const Instance* target_ins, HNode* H_init, std::mt19937* rng);
  Solution create_initial_solution(const Instance* target_ins, int prefix_cost = 0, int step_offset = 0);
  Solution refine_loop(Solution solution, int prefix_cost = 0, const Instance* target_ins = nullptr);

  ScatterBuckets get_rollout_stats(const RolloutResult& rollout, const Scatter& scatter);
  void print_margin_stats() const;

  static constexpr int NUM_SCATTERS = 1;
  static constexpr int NUM_ROLLOUTS = 80;
  static constexpr int STEPS = 8;

  // Pre-allocated stats indexed by margin value (0..100)
  std::vector<ScatterBuckets> margin_stats;
};
