#include "../include/pibt_planner.hpp"
#include "../include/pibt.hpp"
#include "../include/node.hpp"
#include "../include/post_processing.hpp"
#include "../include/refiner.hpp"
#include "../include/metrics.hpp"

#include <iostream>
#include <limits>
#include <chrono>
#include <future>
#include <list>
#include <mutex>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

static constexpr int NUM_OF_THREADS = 7;
static constexpr std::chrono::seconds TIME_ZERO{0};

static bool kbhit()
{
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  int oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
  int ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);
  // Character is already consumed; do not put it back so subsequent calls
  // don't see a stale keypress.
  return ch != EOF;
}

PIBTPlanner::PIBTPlanner(
    const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
    const int _verbose, const Objective _objective, const float _restart_rate)
    : Planner(_ins, _deadline, _MT, _verbose, _objective, _restart_rate)
{
}

// Hierarchical search for the best scatter margin.
// Uses `rng` for all rollouts to ensure reproducibility.
int PIBTPlanner::find_scatter_margin(const Instance* target_ins, HNode* H_init, std::mt19937* rng)
{
  constexpr int MARGIN_ROLLOUTS = 25;

  // Evaluate a given margin: run MARGIN_ROLLOUTS rollouts, return best cost.
  auto eval_margin = [&](int margin) -> uint {
    Scatter sc(target_ins, &D, deadline, 0 /*seed unused*/, 0, margin);
    sc.construct(5);
    PIBT pibt_inst(target_ins, D, rng, &sc);
    uint best = std::numeric_limits<uint>::max();
    for (int r = 0; r < MARGIN_ROLLOUTS; ++r) {
      if (is_expired(deadline)) break;
      auto res = pibt_inst.rollout(H_init);
      if (res.success && res.cost < best) best = res.cost;
    }
    return best;
  };

  // One round of search: given candidates, return the best margin value.
  auto best_of = [&](const std::vector<int>& candidates) -> int {
    int best_margin = candidates[0];
    uint best_cost = std::numeric_limits<uint>::max();
    for (int m : candidates) {
      uint cost = eval_margin(m);
      if (cost < best_cost) { best_cost = cost; best_margin = m; }
    }
    return best_margin;
  };

  // Round 1: coarse search 0..100 step 20
  std::vector<int> coarse;
  for (int m = 0; m <= 100; m += 20) coarse.push_back(m);
  int winner = best_of(coarse);

  // Round 2: step 10 around winner
  {
    std::vector<int> mid;
    for (int m = std::max(0, winner - 10); m <= std::min(100, winner + 10); m += 10)
      mid.push_back(m);
    winner = best_of(mid);
  }

  // Round 3: step 5 around winner
  {
    std::vector<int> fine;
    for (int m = std::max(0, winner - 5); m <= std::min(100, winner + 5); m += 5)
      fine.push_back(m);
    winner = best_of(fine);
  }

  // Round 4: step 2 around winner
  {
    std::vector<int> finest;
    for (int m = std::max(0, winner - 2); m <= std::min(100, winner + 2); m += 2)
      finest.push_back(m);
    winner = best_of(finest);
  }

  return winner;
}

Solution PIBTPlanner::create_initial_solution(const Instance* target_ins, int prefix_cost, int step_offset)
{
  const int local_rollouts = NUM_ROLLOUTS / NUM_OF_THREADS;
  const int base_seed = (MT != nullptr) ? static_cast<int>((*MT)()) : 1;

  auto H_init = new HNode(target_ins->starts, D, nullptr, 0, 0);

  // const int best_margin = find_scatter_margin(target_ins, H_init, MT);
  const int best_margin = 63; // find_scatter_margin(target_ins, H_init, MT);
  // Scatter scatter(target_ins, &D, deadline, base_seed, 0, best_margin);
  // scatter.construct();
  // std::mt19937 rng(base_seed);
  std::cout << "[PIBTPlanner] offset=" << step_offset
            << " best_margin=" << best_margin << std::endl;

  // constexpr int NUM_SCATTERS = 7;
  // std::vector<Scatter> scatters;
  // scatters.reserve(NUM_SCATTERS);
  // for (int s = 0; s < NUM_SCATTERS; ++s) {
  //   scatters.emplace_back(target_ins, &D, deadline, 0, s, best_margin);
  //   scatters.back().construct();
  // }

  std::mutex result_mutex;
  RolloutResult global_best;
  global_best.success = false;
  global_best.cost = std::numeric_limits<uint>::max();

  auto thread_task = [&](int thread_id) {
    std::mt19937 rng(base_seed + thread_id);
    Scatter scatter(target_ins, &D, deadline, base_seed + thread_id, 0, best_margin);
    int scatter_iterations = 50;
    scatter.construct(scatter_iterations);

    PIBT pibt_inst(target_ins, D, &rng, &scatter);
    const bool is_printer = (thread_id == 0);

    RolloutResult local_best;
    local_best.success = false;
    local_best.cost = std::numeric_limits<uint>::max();

    for (int r = 0; r < local_rollouts; ++r) {
      if (is_expired(deadline)) break;

      auto result = pibt_inst.rollout(H_init);

      if (result.success && result.cost < local_best.cost) {
        local_best = std::move(result);

        {
          std::lock_guard<std::mutex> lock(result_mutex);
          if (local_best.cost < global_best.cost) {
            global_best = std::move(local_best);
            std::cout << "[PIBTPlanner] thread: " << thread_id << "  best_cost=" << global_best.cost << std::endl;
          }
        }

      }

      if (is_printer) {
        {
          std::lock_guard<std::mutex> lock(result_mutex);
          int pct = (r + 1) * 100 / local_rollouts;
          std::cout << "[PIBT Planner] "
                    << " progress=" << pct << "%"
                    << " best_cost=" << (global_best.success ? std::to_string(global_best.cost) : "n/a")
                    << "    " << std::endl;
        }
      }
    }

  };

  std::vector<std::future<void>> futures;
  futures.reserve(NUM_OF_THREADS);
  for (int i = 0; i < NUM_OF_THREADS; ++i)
    futures.emplace_back(std::async(std::launch::async, thread_task, i));
  for (auto& f : futures) f.wait();

  std::cout << std::endl;
  delete H_init;

  if (!global_best.success) return Solution{};
  return global_best.configs;
}

Solution PIBTPlanner::refine_loop(Solution solution, int prefix_cost, const Instance* target_ins)
{
  if (solution.empty()) return solution;
  if (target_ins == nullptr) target_ins = ins;

  auto best_cost = get_sum_of_loss(solution, ins->goals);
  std::cout << "[PIBTPlanner] Starting refinement, initial cost: " << (prefix_cost + best_cost) << std::endl;

  int seed_refiner = 0;
  auto last_print = std::chrono::steady_clock::now();

  // Helper to spawn one async refine task using the current best solution.
  auto spawn_refine = [&]() -> std::future<Solution> {
    const int s = (MT != nullptr) ? static_cast<int>((*MT)()) : (++seed_refiner);
    Solution snap = solution;  // capture snapshot for this task
    return std::async(std::launch::async, [=, &D = this->D]() mutable {
      return refine(target_ins, nullptr, snap, &D, s, -1);
    });
  };

  std::list<std::future<Solution>> refiner_pool;
  for (int i = 0; i < NUM_OF_THREADS; ++i)
    refiner_pool.emplace_back(spawn_refine());

  while (!is_expired(deadline)) {
    if (kbhit()) {
      std::cout << "[PIBTPlanner] Key pressed, stopping refinement." << std::endl;
      break;
    }

    refiner_pool.remove_if([&](auto& proc) {
      if (proc.wait_for(TIME_ZERO) != std::future_status::ready) return false;
      auto new_solution = proc.get();
      if (!new_solution.empty()) {
        auto new_cost = get_sum_of_loss(new_solution, ins->goals);
        if (new_cost < best_cost) {
          best_cost = new_cost;
          solution = new_solution;

          auto now = std::chrono::steady_clock::now();
          if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 1) {
            std::cout << "[PIBTPlanner] Refinement cost update -> " << (prefix_cost + best_cost) << std::endl;
            last_print = now;
          }
        }
      }
      if (!is_expired(deadline))
        refiner_pool.emplace_back(spawn_refine());
      return true;
    });
  }

  // Wait for all in-flight tasks to finish before returning.
  for (auto& proc : refiner_pool) proc.wait();

  return solution;
}

Solution PIBTPlanner::solve(std::string& additional_info)
{
  auto solution = create_initial_solution(ins, 0, 0);
  solution = refine_loop(std::move(solution), 0);
  return solution;
}
