/*
 * CEMPlanner: builds an initial policy from random PIBT rollouts,
 * then runs one PolicyPIBT rollout guided by that policy and returns the result.
 */
#pragma once

#include "planner.hpp"
#include "policy.hpp"
#include "policy_pibt.hpp"
#include "rollout_result.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

class CEMPlanner : public Planner {
 public:
  CEMPlanner(const Instance* _ins, const Deadline* _deadline,
                             std::mt19937* _MT, const int _verbose = 0,
                             const Objective _objective = OBJ_NONE,
                             const float _restart_rate = 0.001);

  // Build a NeighborScorePolicy from random PIBT rollouts starting at ins->starts.
  // Runs up to num_rollouts rollouts using this->pibt, keeps the best ones by cost.
  ScorePolicy create_initial_policy(int num_agents,
                                            uint num_rollouts = 5000,
                                            uint keep = 100);

  Solution solve(std::string& additional_info);

  // The best policy found during solve(). Set after solve() returns.
  std::shared_ptr<ScorePolicy> policy;

 private:
  // Run up to num_rollouts random rollouts from H via this->pibt.
  // Returns the RolloutResults of the best `keep` rollouts (by cost).
  std::vector<RolloutResult> get_rollouts(HNode* H,
                                          uint num_rollouts,
                                          uint keep);

  // Generate and evaluate num_candidates PolicyPIBT candidates in parallel,
  // sampling discrete policies from prob_policy. Returns only successful results.
  std::vector<RolloutResult> run_candidate_rollouts(
      const ProbabilityPolicy& prob_policy,
      PolicyRandomizer& randomizer,
      std::vector<std::mt19937>& thread_rngs,
      uint num_candidates);

  // Sort results by cost, truncate to elite_count, and update best_cost /
  // best_configs if the top result improves on the current best.
  void select_elite(std::vector<RolloutResult>& results,
                    uint elite_count,
                    uint& best_cost,
                    std::vector<Config>& best_configs);

  // Build a ScorePolicy by accumulating moves from a set of rollouts.
  // Shared by create_initial_policy and update_policy_with_elite.
  ScorePolicy build_score_policy_from_rollouts(const std::vector<RolloutResult>& rollouts);

  // Smooth prob_policy toward the elite frequencies using the ALPHA blend factor.
  void update_policy_with_elite(ProbabilityPolicy& prob_policy,
                                const std::vector<RolloutResult>& elite);

  // Interactive stall-simulation test. Loops until the user enters 'q' or empty
  // input. Each iteration: samples a discrete policy from prob_policy, lets the
  // user choose N agents to stall, simulates one PIBT step then tries to revert
  // those agents, runs a rollout from the resulting config, and reports feasibility
  // and sum-of-loss.
  void run_stall_test(const ProbabilityPolicy& prob_policy);
};
