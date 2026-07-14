/*
 * PIBTPlanner: runs 10 Scatter instances with different random seeds,
 * one PIBT per Scatter, 10 rollouts each. Returns the globally best rollout.
 */
#pragma once

#include "planner.hpp"
#include "rollout_result.hpp"
#include "scatter.hpp"
#include <memory>
#include <vector>

class PIBTPlanner : public Planner {
 public:
  PIBTPlanner(const Instance* _ins, const Deadline* _deadline,
              std::mt19937* _MT, const int _verbose = 0,
              const Objective _objective = OBJ_NONE,
              const float _restart_rate = 0.001);

  Solution solve(std::string& additional_info);

 private:
  Solution create_initial_solution();
  Solution refine_loop(Solution solution);

  static constexpr int NUM_SCATTERS = 5;
  static constexpr int NUM_ROLLOUTS = 700;
};
