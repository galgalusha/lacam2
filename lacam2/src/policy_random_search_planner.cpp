#include "../include/policy_random_search_planner.hpp"
#include "../include/pibt.hpp"

#include <algorithm>
#include <climits>
#include <future>
#include <iostream>
#include <unordered_map>

static const uint CEM_NUM_CANDIDATES = 100;
static const uint CEM_ELITE_COUNT   = 10;
static const double ALPHA           = 0.2;


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
//  policy.finish_recording(ins);
  return policy;
}

Solution PolicyRandomSearchPlanner::solve_deprecated(std::string& additional_info)
{
  // Build the initial policy from random rollouts.
  auto current_policy = std::make_shared<NeighborScorePolicy>(
      create_initial_policy(ins->N, 5000, 100));

  std::cout << "Initial policy created" << std::endl;

  // Helper: run one PolicyPIBT rollout from starts using the given policy.
  auto run_rollout = [&](std::shared_ptr<NeighborScorePolicy> policy) {
    PolicyPIBT pp(ins, D, policy);
    auto* H = new HNode(ins->starts, D, nullptr, 0, 0);
    auto res = pp.rollout(H);
    delete H;
    return res;
  };

  std::cout << "Running baseline rollout" << std::endl;
  // Establish baseline cost with the initial policy.
  auto baseline = run_rollout(current_policy);
  uint best_cost = baseline.success ? baseline.cost : UINT_MAX;
  std::cout << "Baseline cost: " << best_cost << std::endl;
  std::vector<Config> best_configs = baseline.configs;

  if (baseline.success)
    std::cout << "PolicyRandomSearchPlanner: initial SoC=" << best_cost << std::endl;
  else
    std::cout << "PolicyRandomSearchPlanner: initial rollout failed." << std::endl;

  // Per-thread RNGs for mutation.
  std::uniform_int_distribution<uint> agent_dist(0, ins->N - 1);
  std::uniform_int_distribution<uint> group_size_dist(1, 5);

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

    // Pick a random group of distinct agents (size 1..15).
    uint group_size = std::min(group_size_dist(*MT), (uint)ins->N);
    std::vector<uint> group;
    group.reserve(group_size);
    while (group.size() < group_size) {
      uint idx = agent_dist(*MT);
      if (std::find(group.begin(), group.end(), idx) == group.end())
        group.push_back(idx);
    }

    // Build one candidate: copy of current policy with all group agents randomized.
    auto candidate = std::make_shared<NeighborScorePolicy>(*current_policy);
    for (uint agent_idx : group) {
      candidate->randomize_agent_blind_scores(agent_idx, ins, MT);
      candidate->randomize_agent_scores(agent_idx, MT);
    }

    // Run a single rollout with the candidate policy.
    auto res = run_rollout(candidate);
    if (res.success && res.cost < best_cost) {
      best_cost = res.cost;
      best_configs = res.configs;
      current_policy = candidate;
      std::cout << "PolicyRandomSearchPlanner: new best SoC=" << best_cost
                << " (group_size=" << group_size << ")" << std::endl;
    }
  }

  if (best_configs.empty()) return Solution{};

  policy = current_policy;

  Solution solution;
  solution.push_back(ins->starts);
  for (auto& cfg : best_configs) solution.push_back(cfg);
  return solution;
}

// ---------------------------------------------------------------------------
// CEM-based solve
// ---------------------------------------------------------------------------


Solution PolicyRandomSearchPlanner::solve(std::string& additional_info)
{
  // 1. Build initial policy from random rollouts.
  auto initial_nsp = create_initial_policy(ins->N, 5000, 100);

  // 2. Translate to a ProbabilityPolicy (one AgentProbabilityPolicy per agent).
  AgentPolicyRandomizer randomizer;
  ProbabilityPolicy prob_policy(ins->N);
  const auto& agent_pols = initial_nsp.get_policies();
  for (uint a = 0; a < static_cast<uint>(ins->N); ++a) {
    if (a < agent_pols.size())
      prob_policy[a] = to_probability_policy(agent_pols[a]);
  }

  uint best_cost = UINT_MAX;
  std::vector<Config> best_configs;

  // Per-thread RNGs (used for discrete policy generation and blind-spot sampling).
  std::vector<std::mt19937> thread_rngs(PRS_NUM_THREADS);
  if (MT != nullptr)
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) thread_rngs[i].seed((*MT)());
  else
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) thread_rngs[i].seed(i + 1);

  for (uint gen = 0; !is_expired(deadline); ++gen) {
    // 3-4. Generate and evaluate CEM_NUM_CANDIDATES candidates in parallel.
    //      Each thread copies prob_policy, generates a discrete policy, runs
    //      a PolicyPIBT rollout, and returns the updated discrete+prob policies.
    struct EvalResult {
      uint cost;
      bool success;
      std::vector<Config> configs;
      std::vector<AgentDiscretePolicy> discrete;
      ProbabilityPolicy probs;
    };

    std::vector<EvalResult> eval_results;
    eval_results.reserve(CEM_NUM_CANDIDATES);

    uint dispatched = 0;
    while (dispatched < CEM_NUM_CANDIDATES && !is_expired(deadline)) {
      const uint remaining  = CEM_NUM_CANDIDATES - dispatched;
      const uint batch_size = std::min(remaining, PRS_NUM_THREADS);

      std::vector<std::future<EvalResult>> futures;
      futures.reserve(batch_size);
      for (uint i = 0; i < batch_size; ++i) {
        futures.push_back(std::async(std::launch::async,
            [this, &prob_policy, &randomizer, &thread_rngs, i]() -> EvalResult {
              // Each candidate gets its own copy of the probability policy.
              auto candidate_probs = prob_policy;
              std::vector<AgentDiscretePolicy> disc(ins->N);
              for (uint a = 0; a < static_cast<uint>(ins->N); ++a)
                disc[a] = randomizer(candidate_probs[a], ins, &thread_rngs[i]);

              auto ce_pol = std::make_shared<CrossEntropyPolicy>(
                  std::move(disc), std::move(candidate_probs), ins, &thread_rngs[i]);
              PolicyPIBT pp(ins, D, ce_pol);
              auto* H = new HNode(ins->starts, D, nullptr, 0, 0);
              auto res = pp.rollout(H);
              delete H;
              return {res.cost, res.success, std::move(res.configs),
                      std::move(ce_pol->discrete), std::move(ce_pol->probs)};
            }));
      }

      for (auto& f : futures) {
        auto er = f.get();
        if (er.success) eval_results.push_back(std::move(er));
      }
      dispatched += batch_size;
    }

    // 5. Sort by cost, keep best CEM_ELITE_COUNT.
    std::sort(eval_results.begin(), eval_results.end(),
        [](const EvalResult& a, const EvalResult& b) { return a.cost < b.cost; });
    if (eval_results.size() > CEM_ELITE_COUNT) eval_results.resize(CEM_ELITE_COUNT);

    if (!eval_results.empty() && eval_results[0].cost < best_cost) {
      best_cost    = eval_results[0].cost;
      best_configs = eval_results[0].configs;
    }

    std::cout << "PolicyRandomSearchPlanner CEM gen=" << gen
              << " elite=" << eval_results.size()
              << " best_SoC=" << best_cost << std::endl;

    if (eval_results.empty()) continue;

    // 6. Smoothed probability update using elite frequencies.
    for (uint a = 0; a < static_cast<uint>(ins->N); ++a) {
      auto& master_agent = prob_policy[a];

      // Collect elite frequencies across all elite results.
      std::unordered_map<Vertex*, std::unordered_map<Vertex*, uint>> neighbor_counts;
      std::unordered_map<Vertex*, uint> E_v_map;
      for (const auto& er : eval_results) {
        for (const auto& [v, fav] : er.discrete[a].favorite) {
          ++E_v_map[v];
          ++neighbor_counts[v][fav];
        }
        // Merge vertices discovered lazily during this elite rollout into master.
        for (const auto& [v, nb_probs] : er.probs[a].vertex_probs) {
          if (!master_agent.vertex_probs.count(v))
            master_agent.vertex_probs[v] = nb_probs;  // uniform initial value
        }
      }

      // Apply smoothed update to every vertex in master that had elite activity.
      for (auto& [v, nb_probs] : master_agent.vertex_probs) {
        auto ev_it = E_v_map.find(v);
        if (ev_it == E_v_map.end() || ev_it->second == 0) continue;
        const uint E_v = ev_it->second;
        const auto& vcnt = neighbor_counts[v];
        for (auto& [u, p_old] : nb_probs) {
          auto cnt_it = vcnt.find(u);
          const double N_u = cnt_it != vcnt.end() ? cnt_it->second : 0.0;
          const double p_elite = N_u / E_v;
          p_old = (1.0 - ALPHA) * p_old + ALPHA * p_elite;
        }
      }
    }
  }

  if (best_configs.empty()) return Solution{};

  Solution solution;
  solution.push_back(ins->starts);
  for (auto& cfg : best_configs) solution.push_back(cfg);
  return solution;
}
