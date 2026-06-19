/*
 * RandomPlanner implementation.
 */
#include "../include/random_planner.hpp"

#include <algorithm>
#include <iostream>
#include <limits>

RandomPlanner::RandomPlanner(const Instance* _ins, const Deadline* _deadline,
                             std::mt19937* _MT, int _verbose)
    : ins(_ins), deadline(_deadline), MT(_MT), verbose(_verbose), D(_ins)
{
}

Solution RandomPlanner::solve()
{
  // Build a single HNode for the initial configuration.  We reuse it for every
  // rollout so we never allocate search-tree nodes.
  DistTable& D_ref = D;
  HNode* start_node = new HNode(ins->starts, D_ref, nullptr, 0,
                                /*h=*/0);

  // Randomise the PIBT priority order each rollout by shuffling the order
  // vector stored in the start node.  PIBT::rollout reads H->order.
  PIBT pibt(ins, D_ref, MT);

  uint best_cost = std::numeric_limits<uint>::max();
  Solution best_solution;
  uint rollout_count = 0;

  const uint N = ins->N;

  // Base priorities by distance to goal — computed once, fixed across rollouts.
  std::vector<float> base_priorities(N);
  for (uint i = 0; i < N; ++i)
    base_priorities[i] = (float)D.get(i, start_node->C[i]);

  while (!is_expired(deadline)) {
    // Add a small random perturbation to break ties differently each rollout.
    for (uint i = 0; i < N; ++i)
      start_node->priorities[i] = base_priorities[i] + get_random_float(MT) * 2;

    // Re-sort order according to updated priorities (descending).
    std::sort(start_node->order.begin(), start_node->order.end(),
              [&](uint a, uint b) {
                return start_node->priorities[a] > start_node->priorities[b];
              });

    auto result = pibt.rollout(start_node);
    ++rollout_count;

    if (result.success && result.cost < best_cost) {
      best_cost = result.cost;

      // Reconstruct solution: starts config + all successor configs.
      best_solution.clear();
      best_solution.push_back(ins->starts);
      for (const auto& cfg : result.configs) best_solution.push_back(cfg);

      info(0, verbose, "RandomPlanner: rollout #", rollout_count,
           "  new best cost=", best_cost,
           "  makespan=", result.makespan,
           "  elapsed=", elapsed_ms(deadline), "ms");
    }
  }

  delete start_node;

  info(1, verbose, "RandomPlanner: total rollouts=", rollout_count,
       "  best_cost=", best_cost);

  return best_solution;
}
