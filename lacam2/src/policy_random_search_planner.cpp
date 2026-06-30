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
  return NeighborScorePolicy(std::move(agent_policies), MT);
}

Solution PolicyRandomSearchPlanner::solve(std::string& additional_info)
{
  // Build the initial policy from random rollouts.
  auto policy = std::make_shared<NeighborScorePolicy>(
      create_initial_policy(ins->N));

  // Run one PolicyPIBT rollout from the start using the policy.
  PolicyPIBT policy_pibt(ins, D, policy);
  auto H_init = new HNode(ins->starts, D, nullptr, 0, 0);

  std::cout << "PolicyRandomSearchPlanner: running policy rollout..." << std::endl;
  auto res = policy_pibt.rollout(H_init);
  delete H_init;

  if (!res.success || res.configs.empty()) {
    std::cout << "PolicyRandomSearchPlanner: policy rollout failed." << std::endl;
    return Solution{};
  }

  std::cout << "PolicyRandomSearchPlanner: policy rollout succeeded. Cost="
            << res.cost << std::endl;

  // Build solution: starts config + rollout configs.
  Solution solution;
  solution.push_back(ins->starts);
  for (auto& cfg : res.configs) solution.push_back(cfg);
  return solution;
}
