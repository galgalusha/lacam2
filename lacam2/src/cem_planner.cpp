#include "../include/cem_planner.hpp"
#include "../include/pibt.hpp"
#include "../include/post_processing.hpp"

#include <algorithm>
#include <climits>
#include <future>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

static bool key_pressed_to_stop()
{
  // Non-blocking check for Space (0x20) or Escape (0x1B) on stdin.
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, old_flags | O_NONBLOCK);

  char c = 0;
  bool stop = false;
  if (read(STDIN_FILENO, &c, 1) == 1)
    stop = (c == ' ' || c == 0x1B);

  fcntl(STDIN_FILENO, F_SETFL, old_flags);
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return stop;
}

static const uint CEM_NUM_CANDIDATES = 100;
static const uint CEM_ELITE_COUNT   = 10;
static const double ALPHA           = 0.2;


CEMPlanner::CEMPlanner(
    const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
    const int _verbose, const Objective _objective, const float _restart_rate)
    : Planner(_ins, _deadline, _MT, _verbose, _objective, _restart_rate)
{
}

static const uint PRS_NUM_THREADS = 7;

std::vector<std::vector<Config>> CEMPlanner::get_rollouts(
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

NeighborScorePolicy CEMPlanner::create_initial_policy(
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

Solution CEMPlanner::solve_deprecated(std::string& additional_info)
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
// CEM helper methods
// ---------------------------------------------------------------------------

std::vector<CEMPlanner::EvalResult> CEMPlanner::run_candidate_rollouts(
    const ProbabilityPolicy& prob_policy,
    AgentPolicyRandomizer& randomizer,
    std::vector<std::mt19937>& thread_rngs,
    uint num_candidates)
{
  std::vector<EvalResult> eval_results;
  eval_results.reserve(num_candidates);

  uint dispatched = 0;
  while (dispatched < num_candidates && !is_expired(deadline)) {
    const uint remaining  = num_candidates - dispatched;
    const uint batch_size = std::min(remaining, PRS_NUM_THREADS);

    std::vector<std::future<EvalResult>> futures;
    futures.reserve(batch_size);
    for (uint i = 0; i < batch_size; ++i) {
      futures.push_back(std::async(std::launch::async,
          [this, &prob_policy, &randomizer, &thread_rngs, i]() -> EvalResult {
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

  return eval_results;
}

void CEMPlanner::select_elite(std::vector<EvalResult>& eval_results,
                              uint elite_count,
                              uint& best_cost,
                              std::vector<Config>& best_configs)
{
  std::sort(eval_results.begin(), eval_results.end(),
      [](const EvalResult& a, const EvalResult& b) { return a.cost < b.cost; });
  if (eval_results.size() > elite_count) eval_results.resize(elite_count);

  if (!eval_results.empty() && eval_results[0].cost < best_cost) {
    best_cost    = eval_results[0].cost;
    best_configs = eval_results[0].configs;
  }
}

void CEMPlanner::update_policy_with_elite(ProbabilityPolicy& prob_policy,
                                          const std::vector<EvalResult>& elite)
{
  for (uint a = 0; a < static_cast<uint>(ins->N); ++a) {
    auto& master_agent = prob_policy[a];

    std::unordered_map<Vertex*, std::unordered_map<Vertex*, uint>> neighbor_counts;
    std::unordered_map<Vertex*, uint> E_v_map;
    for (const auto& er : elite) {
      for (const auto& [v, fav] : er.discrete[a].favorite) {
        ++E_v_map[v];
        ++neighbor_counts[v][fav];
      }
      for (const auto& [v, nb_probs] : er.probs[a].vertex_probs) {
        if (!master_agent.vertex_probs.count(v))
          master_agent.vertex_probs[v] = nb_probs;
      }
    }

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

// ---------------------------------------------------------------------------
// CEM-based solve
// ---------------------------------------------------------------------------

Solution CEMPlanner::solve(std::string& additional_info)
{
  // 1. Build initial policy from random rollouts.
  auto initial_nsp = create_initial_policy(ins->N, 5000, 100);

  // 2. Translate to a ProbabilityPolicy.
  AgentPolicyRandomizer randomizer;
  ProbabilityPolicy prob_policy(ins->N);
  const auto& agent_pols = initial_nsp.get_policies();
  for (uint a = 0; a < static_cast<uint>(ins->N); ++a) {
    if (a < agent_pols.size())
      prob_policy[a] = to_probability_policy(agent_pols[a]);
  }

  uint best_cost = UINT_MAX;
  std::vector<Config> best_configs;

  std::vector<std::mt19937> thread_rngs(PRS_NUM_THREADS);
  if (MT != nullptr)
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) thread_rngs[i].seed((*MT)());
  else
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) thread_rngs[i].seed(i + 1);

  for (uint gen = 0; !is_expired(deadline); ++gen) {
    if (key_pressed_to_stop()) {
      std::cout << "CEMPlanner: interrupted by user (Space/Escape)." << std::endl;
      break;
    }
    // 3-4. Generate and evaluate candidates.
    auto eval_results = run_candidate_rollouts(
        prob_policy, randomizer, thread_rngs, CEM_NUM_CANDIDATES);

    // 5. Select elite.
    select_elite(eval_results, CEM_ELITE_COUNT, best_cost, best_configs);

    std::cout << "PolicyRandomSearchPlanner CEM gen=" << gen
              << " elite=" << eval_results.size()
              << " best_SoC=" << best_cost << std::endl;

    if (eval_results.empty()) continue;

    // 6. Smooth probability update toward elite.
    update_policy_with_elite(prob_policy, eval_results);
  }

  if (best_configs.empty()) return Solution{};

  run_stall_test(prob_policy);

  Solution solution;
  solution.push_back(ins->starts);
  for (auto& cfg : best_configs) solution.push_back(cfg);
  return solution;
}

// ---------------------------------------------------------------------------
// Interactive stall-simulation test
// ---------------------------------------------------------------------------

void CEMPlanner::run_stall_test(const ProbabilityPolicy& prob_policy)
{
  std::cout << "\n=== Stall-simulation test (q / empty Enter to quit) ===" << std::endl;

  AgentPolicyRandomizer randomizer;

  // Restore blocking terminal I/O for interactive use.
  struct termios oldt;
  tcgetattr(STDIN_FILENO, &oldt);
  struct termios newt = oldt;
  newt.c_lflag |= (ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, old_flags & ~O_NONBLOCK);

  while (true) {
    // --- prompt ---
    std::cout << "\nNumber of agents to stall [0-" << ins->N << "] (q or Enter=quit): ";
    std::string line;
    if (!std::getline(std::cin, line)) break;
    if (line.empty() || line == "q" || line == "Q") break;

    int num_stall = 0;
    try { num_stall = std::stoi(line); }
    catch (...) { std::cout << "Invalid input, try again." << std::endl; continue; }
    if (num_stall < 0 || num_stall > ins->N) {
      std::cout << "Out of range, try again." << std::endl; continue;
    }

    // --- 1. sample a discrete policy from the master prob_policy ---
    std::vector<AgentDiscretePolicy> disc(ins->N);
    for (uint a = 0; a < static_cast<uint>(ins->N); ++a)
      disc[a] = randomizer(prob_policy[a], ins, MT);
    auto ce_pol = std::make_shared<CrossEntropyPolicy>(
        std::move(disc), prob_policy, ins, MT);

    // --- 2. pick random stall agents ---
    std::vector<uint> stall_agents;
    {
      std::vector<uint> all(ins->N);
      std::iota(all.begin(), all.end(), 0);
      std::shuffle(all.begin(), all.end(), *MT);
      stall_agents.assign(all.begin(), all.begin() + num_stall);
    }
    std::cout << "Stalling agents:";
    for (auto a : stall_agents) std::cout << " " << a;
    std::cout << std::endl;

    // --- 3. run one PIBT step from starts ---
    PolicyPIBT pp_step(ins, D, ce_pol);
    auto* H_start = new HNode(ins->starts, D, nullptr, 0, 0);
    LNode unconstrained;
    Config C_step1(ins->N, nullptr);
    if (!pp_step.get_new_config(H_start, &unconstrained, C_step1)) {
      std::cout << "PIBT step failed, skipping." << std::endl;
      delete H_start;
      continue;
    }

    // --- 4. try to revert stall agents to their starts position ---
    // Build a set of vertices occupied in C_step1 (excluding stall agents,
    // since they are the candidates to be reverted).
    std::unordered_set<uint> stall_set(stall_agents.begin(), stall_agents.end());
    std::unordered_map<uint /*vertex id*/, uint /*agent*/> occupied_after;
    for (uint a = 0; a < static_cast<uint>(ins->N); ++a) {
      if (!stall_set.count(a))
        occupied_after[C_step1[a]->id] = a;
    }

    int intended_reverts  = num_stall;
    int successful_reverts = 0;
    Config C_stalled = C_step1;  // copy; we'll overwrite stall agents if possible
    for (uint a : stall_agents) {
      Vertex* prev = ins->starts[a];
      if (!occupied_after.count(prev->id)) {
        // vertex is free — revert
        C_stalled[a] = prev;
        occupied_after[prev->id] = a;  // mark as now occupied
        ++successful_reverts;
      }
    }
    std::cout << "Reverts: " << successful_reverts << "/" << intended_reverts << std::endl;

    // --- 5. rollout from stalled config ---
    // Generate 20 discrete policies in parallel, pick the best successful rollout.
    static constexpr uint STALL_NUM_POLICIES = 20;
    std::vector<std::mt19937> stall_rngs(STALL_NUM_POLICIES);
    for (uint i = 0; i < STALL_NUM_POLICIES; ++i)
      stall_rngs[i].seed(MT ? (*MT)() : i + 42);

    struct StallResult { RolloutResult res; };
    std::vector<std::future<StallResult>> stall_futures;
    stall_futures.reserve(STALL_NUM_POLICIES);
    for (uint i = 0; i < STALL_NUM_POLICIES; ++i) {
      stall_futures.push_back(std::async(std::launch::async,
          [this, &prob_policy, &randomizer, &C_stalled, &stall_rngs, i]() -> StallResult {
            std::vector<AgentDiscretePolicy> d(ins->N);
            for (uint a = 0; a < static_cast<uint>(ins->N); ++a)
              d[a] = randomizer(prob_policy[a], ins, &stall_rngs[i]);
            auto pol = std::make_shared<CrossEntropyPolicy>(
                std::move(d), prob_policy, ins, &stall_rngs[i]);
            PolicyPIBT pp(ins, D, pol);
            auto* H = new HNode(C_stalled, D, nullptr, 0, 0);
            auto r = pp.rollout(H);
            delete H;
            return {std::move(r)};
          }));
    }

    RolloutResult best_res;
    best_res.success = false;
    best_res.cost    = UINT_MAX;
    for (auto& f : stall_futures) {
      auto sr = f.get();
      if (sr.res.success && sr.res.cost < best_res.cost)
        best_res = std::move(sr.res);
    }
    auto& res = best_res;
    delete H_start;

    if (!res.success) {
      std::cout << "Rollout from stalled config did not reach goal." << std::endl;
      continue;
    }

    // --- 6. build full solution: starts -> C_stalled -> rollout ---
    Solution sol;
    sol.push_back(ins->starts);
    sol.push_back(C_stalled);
    for (auto& cfg : res.configs) sol.push_back(cfg);

    // --- 7. feasibility check ---
    bool feasible = is_feasible_solution(*ins, sol, /*verbose=*/1);
    std::cout << "Feasible: " << (feasible ? "yes" : "NO") << std::endl;

    // --- 8. sum of loss ---
    int sum_loss = get_sum_of_loss(sol);
    std::cout << "Sum-of-loss: " << sum_loss << std::endl;
  }

  // Restore terminal to its original state.
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, old_flags);
  std::cout << "=== Stall test finished ===" << std::endl;
}
