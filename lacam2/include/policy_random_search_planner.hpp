/*
 * PolicyRandomSearchPlanner: builds an initial policy from random PIBT rollouts,
 * then runs one PolicyPIBT rollout guided by that policy and returns the result.
 */
#pragma once

#include "planner.hpp"
#include "policy.hpp"
#include "policy_pibt.hpp"
#include "rollout_result.hpp"

#include <memory>
#include <vector>

class PolicyRandomSearchPlanner : public Planner {
 public:
  PolicyRandomSearchPlanner(const Instance* _ins, const Deadline* _deadline,
                             std::mt19937* _MT, const int _verbose = 0,
                             const Objective _objective = OBJ_NONE,
                             const float _restart_rate = 0.001);

  // Build a NeighborScorePolicy from random PIBT rollouts starting at ins->starts.
  // Runs up to num_rollouts rollouts using this->pibt, keeps the best ones by cost.
  NeighborScorePolicy create_initial_policy(int num_agents,
                                            uint num_rollouts = 5000,
                                            uint keep = 100);

  Solution solve(std::string& additional_info);
  Solution solve_deprecated(std::string& additional_info);

  // The best policy found during solve(). Set after solve() returns.
  std::shared_ptr<NeighborScorePolicy> policy;

 private:
  // Run up to num_rollouts random rollouts from H via this->pibt.
  // Returns the configs sequences of the best `keep` rollouts (by cost).
  std::vector<std::vector<Config>> get_rollouts(HNode* H,
                                                uint num_rollouts,
                                                uint keep);
};
