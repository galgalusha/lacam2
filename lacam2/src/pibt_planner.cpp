#include "../include/pibt_planner.hpp"
#include "../include/pibt.hpp"
#include "../include/node.hpp"
#include "../include/post_processing.hpp"
#include "../include/refiner.hpp"
#include "../include/metrics.hpp"

#include <iostream>
#include <limits>

PIBTPlanner::PIBTPlanner(
    const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
    const int _verbose, const Objective _objective, const float _restart_rate)
    : Planner(_ins, _deadline, _MT, _verbose, _objective, _restart_rate)
{
}

Solution PIBTPlanner::create_initial_solution()
{
  RolloutResult best;
  best.success = false;
  best.cost = std::numeric_limits<uint>::max();

  auto H_init = new HNode(ins->starts, D, nullptr, 0, 0);

  for (int s = 0; s < NUM_SCATTERS; ++s) {
    std::cout << "[PIBTPlanner] Running PIBT " << s+1 << " out of " << NUM_SCATTERS << std::endl;
    const int seed = (MT != nullptr) ? static_cast<int>((*MT)()) : (s + 1);

    Scatter scatter_inst(ins, &D, deadline, seed, 4, 10);
    scatter_inst.construct();

    std::mt19937 rng(seed);
    PIBT pibt_inst(ins, D, &rng, &scatter_inst);

    for (int r = 0; r < NUM_ROLLOUTS; ++r) {
      if (is_expired(deadline)) break;

      auto result = pibt_inst.rollout(H_init);

      if (result.success && result.cost < best.cost) {
        std::cout << "[PIBTPlanner] Cost update " << best.cost << "->" << result.cost << std::endl;
        best = std::move(result);
      }
    }

    if (is_expired(deadline)) break;
  }

  delete H_init;

  if (!best.success) return Solution{};
  return best.configs;
}

Solution PIBTPlanner::refine_loop(Solution solution)
{
  if (solution.empty()) return solution;

  auto best_cost = get_sum_of_loss(solution);
  std::cout << "[PIBTPlanner] Starting refinement, initial cost: " << best_cost << std::endl;

  auto seed = 0;
  while (!is_expired(deadline)) {
    const int refine_seed = (MT != nullptr) ? static_cast<int>((*MT)()) : (++seed);
    auto refined = refine(ins, deadline, solution, &D, refine_seed, verbose);
    if (refined.empty()) continue;

    auto new_cost = get_sum_of_loss(refined);
    if (new_cost < best_cost) {
      std::cout << "[PIBTPlanner] Refinement cost update " << best_cost << "->" << new_cost << std::endl;
      best_cost = new_cost;
      solution = std::move(refined);
    }
  }

  return solution;
}

Solution PIBTPlanner::solve(std::string& additional_info)
{
  auto solution = create_initial_solution();
  solution = refine_loop(std::move(solution));

  additional_info = "pibt_planner";
  return solution;
}
