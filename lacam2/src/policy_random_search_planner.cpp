#include "../include/policy_random_search_planner.hpp"
#include "../include/pibt.hpp"

#include <algorithm>
#include <future>
#include <iostream>


PolicyRandomSearchPlanner::PolicyRandomSearchPlanner(
    const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
    const int _verbose, const Objective _objective, const float _restart_rate)
    : Planner(_ins, _deadline, _MT, _verbose, _objective, _restart_rate)
{
}

static const uint PRS_NUM_THREADS = 7;

std::vector<std::vector<Config>> PolicyRandomSearchPlanner::get_rollouts(
    HNode* H, uint num_rollouts, uint keep)
{
  struct RolloutEntry {
    uint cost;
    std::vector<Config> configs;
  };

  // Create one RNG and one PIBT instance per thread.
  std::vector<std::mt19937> rollout_rngs(PRS_NUM_THREADS);
  if (MT != nullptr) {
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) rollout_rngs[i].seed((*MT)());
  } else {
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) rollout_rngs[i].seed(i + 1);
  }

  std::vector<std::unique_ptr<PIBTBase>> pibts;
  pibts.reserve(PRS_NUM_THREADS);
  for (uint i = 0; i < PRS_NUM_THREADS; ++i)
    pibts.push_back(std::make_unique<PIBT>(ins, D, &rollout_rngs[i]));

  std::vector<RolloutEntry> results;
  results.reserve(num_rollouts);

  uint dispatched = 0;
  while (dispatched < num_rollouts && !is_expired(deadline)) {
    const uint remaining = num_rollouts - dispatched;
    const uint batch_size = std::min(remaining, PRS_NUM_THREADS);

    std::vector<std::future<RolloutResult>> futures;
    futures.reserve(batch_size);
    for (uint i = 0; i < batch_size; ++i) {
      futures.push_back(std::async(std::launch::async,
                                   [&pibts, i, H]() {
                                     return pibts[i]->rollout(H);
                                   }));
    }

    for (auto& f : futures) {
      auto res = f.get();
      if (res.success) results.push_back({res.cost, std::move(res.configs)});
    }

    dispatched += batch_size;
  }

  // Sort by cost ascending, keep the best `keep`.
  std::sort(results.begin(), results.end(),
            [](const RolloutEntry& a, const RolloutEntry& b) {
              return a.cost < b.cost;
            });

  if (results.size() > keep) results.resize(keep);

  std::vector<std::vector<Config>> out;
  out.reserve(results.size());
  for (auto& e : results) out.push_back(std::move(e.configs));
  return out;
}

NeighborScorePolicy PolicyRandomSearchPlanner::create_initial_policy(
    int num_agents, uint num_rollouts, uint keep)
{
  auto H_init = new HNode(ins->starts, D, nullptr, 0, 0);

  std::cout << "PolicyRandomSearchPlanner: generating " << num_rollouts
            << " rollouts for initial policy" << std::endl;

  auto best_rollouts = get_rollouts(H_init, num_rollouts, keep);

  std::cout << "PolicyRandomSearchPlanner: done. Got " << best_rollouts.size()
            << " successful rollouts." << std::endl;

  std::vector<AgentPolicy> agent_policies(num_agents);

  for (const auto& rollout : best_rollouts) {
    for (size_t step = 1; step < rollout.size(); ++step) {
      const Config& prev = rollout[step - 1];
      const Config& curr = rollout[step];
      for (int agent = 0; agent < num_agents; ++agent) {
        Vertex* from = prev[agent];
        Vertex* to   = curr[agent];
        if (from != to) {
          agent_policies[agent].record_move(from, to);
        }
      }
    }
  }

  delete H_init;
  auto policy = NeighborScorePolicy(std::move(agent_policies), MT);
  policy.finish_recording(ins);
  return policy;
}

Solution PolicyRandomSearchPlanner::solve(std::string& additional_info)
{
  // Build the initial policy from random rollouts.
  auto current_policy = std::make_shared<NeighborScorePolicy>(
      create_initial_policy(ins->N));

  // Helper: run one PolicyPIBT rollout from starts using the given policy.
  auto run_rollout = [&](std::shared_ptr<NeighborScorePolicy> policy) {
    PolicyPIBT pp(ins, D, policy);
    auto* H = new HNode(ins->starts, D, nullptr, 0, 0);
    auto res = pp.rollout(H);
    delete H;
    return res;
  };

  // Establish baseline cost with the initial policy.
  auto baseline = run_rollout(current_policy);
  uint best_cost = baseline.success ? baseline.cost : UINT_MAX;
  std::vector<Config> best_configs = baseline.configs;

  if (baseline.success)
    std::cout << "PolicyRandomSearchPlanner: initial SoC=" << best_cost << std::endl;
  else
    std::cout << "PolicyRandomSearchPlanner: initial rollout failed." << std::endl;

  // Per-thread RNGs for mutation.
  std::vector<std::mt19937> rngs(PRS_NUM_THREADS);
  if (MT != nullptr) {
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) rngs[i].seed((*MT)());
  } else {
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) rngs[i].seed(i + 1000);
  }
  std::uniform_int_distribution<uint> agent_dist(0, ins->N - 1);

  auto last_print = std::chrono::steady_clock::now();

  while (!is_expired(deadline)) {
    loop_cnt++;

    // Print iteration count every 2 seconds.
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 2) {
      std::cout << "PolicyRandomSearchPlanner: iterations=" << loop_cnt
                << " best_SoC=" << best_cost << std::endl;
      last_print = now;
    }

    // Pick one agent to mutate this round.
    uint agent_idx = agent_dist(*MT);

    // Create 6 candidates with the agent's blind_score_map re-randomized,
    // plus 1 candidate with that agent's vertex_scores shuffled.
    // Total = PRS_NUM_THREADS = 7.
    const uint total_candidates = PRS_NUM_THREADS;
    std::vector<std::shared_ptr<NeighborScorePolicy>> candidates;
    candidates.reserve(total_candidates);
    for (uint i = 0; i < PRS_NUM_THREADS - 1; ++i) {
      auto candidate = std::make_shared<NeighborScorePolicy>(*current_policy);
      candidate->randomize_agent_blind_scores(agent_idx, ins, &rngs[i]);
      candidates.push_back(std::move(candidate));
    }
    // Last candidate: shuffle vertex_scores for the same agent.
    {
      auto candidate = std::make_shared<NeighborScorePolicy>(*current_policy);
      candidate->randomize_agent_scores(agent_idx, &rngs[PRS_NUM_THREADS - 1]);
      candidates.push_back(std::move(candidate));
    }

    // Run rollouts in parallel.
    std::vector<std::future<RolloutResult>> futures;
    futures.reserve(total_candidates);
    for (uint i = 0; i < total_candidates; ++i) {
      futures.push_back(std::async(std::launch::async,
                                   [this, &candidates, i]() {
                                     PolicyPIBT pp(ins, D, candidates[i]);
                                     auto* H = new HNode(ins->starts, D, nullptr, 0, 0);
                                     auto res = pp.rollout(H);
                                     delete H;
                                     return res;
                                   }));
    }

    // Collect results; adopt any improvement.
    for (uint i = 0; i < total_candidates; ++i) {
      auto res = futures[i].get();
      if (res.success && res.cost < best_cost) {
        best_cost = res.cost;
        best_configs = res.configs;
        current_policy = candidates[i];
        std::cout << "PolicyRandomSearchPlanner: new best SoC=" << best_cost << std::endl;
      }
    }
  }

  if (best_configs.empty()) return Solution{};

  Solution solution;
  solution.push_back(ins->starts);
  for (auto& cfg : best_configs) solution.push_back(cfg);
  return solution;
}
