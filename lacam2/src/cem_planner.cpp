#include "../include/cem_planner.hpp"
#include "../include/cem_ui.hpp"
#include "../include/pibt.hpp"
#include "../include/post_processing.hpp"
#include "../include/cem_params.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <future>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

CEMPlanner::CEMPlanner(
    const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
    const int _verbose, const Objective _objective, const float _restart_rate)
    : Planner(_ins, _deadline, _MT, _verbose, _objective, _restart_rate)
{
}

std::vector<RolloutResult> CEMPlanner::get_rollouts(
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
    pibts.push_back(std::make_unique<PIBT>(ins, D, &rollout_rngs[i], scatter));

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

  std::vector<RolloutResult> out;
  out.reserve(results.size());
  for (auto& e : results) {
    RolloutResult rr;
    rr.success = true;
    rr.cost = e.cost;
    rr.makespan = static_cast<uint>(e.configs.size());
    rr.configs = std::move(e.configs);
    out.push_back(std::move(rr));
  }
  return out;
}

ScorePolicy CEMPlanner::create_initial_policy(
    int num_agents, std::vector<RolloutResult>& global_elite, uint num_rollouts, uint keep)
{
  auto H_init = new HNode(ins->starts, D, nullptr, 0, 0);

  std::cout << "CEMPlanner: generating " << num_rollouts
            << " rollouts for initial policy" << std::endl;

  auto best_rollouts = get_rollouts(H_init, num_rollouts, keep);
  std::cout << "CEMPlanner: done. Got " << best_rollouts.size()
            << " successful rollouts." << std::endl;
  delete H_init;

  // Populate global_elite with the best rollouts (already sorted by cost).
  global_elite = best_rollouts;
  if (static_cast<int>(global_elite.size()) > CEM_ELITE_COUNT)
    global_elite.resize(CEM_ELITE_COUNT);

  return build_score_policy_from_rollouts(best_rollouts);
}

ScorePolicy CEMPlanner::build_score_policy_from_rollouts(
    const std::vector<RolloutResult>& rollouts)
{
  std::vector<AgentScores> agent_policies(ins->N);

  for (const auto& rollout : rollouts) {
    for (size_t step = 1; step < rollout.configs.size(); ++step) {
      const Config& prev = rollout.configs[step - 1];
      const Config& curr = rollout.configs[step];

      for (int agent = 0; agent < ins->N; ++agent) {
        Vertex* from = prev[agent];
        Vertex* to   = curr[agent];
        if (from != to) agent_policies[agent].record_move(from, to);
      }
    }
  }

  return ScorePolicy(std::move(agent_policies), MT);
}

// ---------------------------------------------------------------------------
// CEM helper methodsCEM_ELITE_COUNT
// ---------------------------------------------------------------------------

std::vector<RolloutResult> CEMPlanner::run_candidate_rollouts(
    const ProbabilityPolicy& prob_policy,
    PolicyRandomizer& randomizer,
    std::vector<std::mt19937>& thread_rngs,
    uint num_candidates,
    uint max_cost)
{
  // Generate all candidate policies upfront before spawning threads.
  const auto t_rng_start = std::chrono::steady_clock::now();
  auto policies = randomizer(prob_policy, ins, thread_rngs, num_candidates);
  g_last_randomizer_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_rng_start).count();

  const auto t_rollouts_start = std::chrono::steady_clock::now();

  std::vector<RolloutResult> results;
  results.reserve(num_candidates);

  uint dispatched = 0;
  while (dispatched < static_cast<uint>(policies.size()) && !is_expired(deadline)) {
    const uint remaining  = static_cast<uint>(policies.size()) - dispatched;
    const uint batch_size = std::min(remaining, PRS_NUM_THREADS);

    std::vector<std::future<RolloutResult>> futures;
    futures.reserve(batch_size);
    for (uint i = 0; i < batch_size; ++i) {
      auto pol = policies[dispatched + i];
      futures.push_back(std::async(std::launch::async,
          [this, pol, max_cost]() -> RolloutResult {
            PolicyPIBT pp(ins, D, MT, pol, scatter);
            auto* H = new HNode(ins->starts, D, nullptr, 0, 0);
            auto res = pp.rollout(H, max_cost);
            delete H;
            return res;
          }));
    }

    for (auto& f : futures) {
      auto res = f.get();
      if (res.success) results.push_back(std::move(res));
    }
    dispatched += batch_size;
  }
  g_last_rollouts_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t_rollouts_start).count();

  return results;
}

int CEMPlanner::update_global_elite(std::vector<RolloutResult>& global_elite,
                                    const std::vector<RolloutResult>& new_results)
{
  int new_elite_count = 0;
  for (const auto& rr : new_results) {
    if (static_cast<int>(global_elite.size()) < CEM_ELITE_COUNT) {
      global_elite.push_back(rr);
      std::sort(global_elite.begin(), global_elite.end(),
          [](const RolloutResult& a, const RolloutResult& b) { return a.cost < b.cost; });
      ++new_elite_count;
    } else if (!global_elite.empty() && rr.cost < global_elite.back().cost) {
      global_elite.back() = rr;
      std::sort(global_elite.begin(), global_elite.end(),
          [](const RolloutResult& a, const RolloutResult& b) { return a.cost < b.cost; });
      ++new_elite_count;
    }
  }
  return new_elite_count;
}

void CEMPlanner::select_elite(std::vector<RolloutResult>& results,
                              uint elite_count,
                              uint& best_cost,
                              std::vector<Config>& best_configs)
{
  std::sort(results.begin(), results.end(),
      [](const RolloutResult& a, const RolloutResult& b) { return a.cost < b.cost; });
  if (results.size() > elite_count) results.resize(elite_count);

  if (!results.empty() && results[0].cost < best_cost) {
    best_cost    = results[0].cost;
    best_configs = results[0].configs;
  }
}

void CEMPlanner::update_policy_with_elite(ProbabilityPolicy& prob_policy,
                                          const std::vector<RolloutResult>& elite)
{
  auto score_pol = build_score_policy_from_rollouts(elite);
  const auto& agent_scores = score_pol.get_policies();

  for (uint a = 0; a < static_cast<uint>(ins->N); ++a) {
    auto& master_agent = prob_policy[a];
    if (a >= agent_scores.size()) continue;

    const auto elite_prob = to_probability_policy(agent_scores[a], GEN_LAPLACE_SMOOTHING);

    // Add newly seen vertices to the master policy.
    for (const auto& [v, nb_probs] : elite_prob.vertex_probs) {
      if (!master_agent.vertex_probs.count(v))
        master_agent.vertex_probs[v] = nb_probs;
    }

    // Blend existing entries toward elite frequencies.
    for (auto& [v, nb_probs] : master_agent.vertex_probs) {
      auto eit = elite_prob.vertex_probs.find(v);
      if (eit == elite_prob.vertex_probs.end()) continue;
      const auto& elite_nb = eit->second;
      for (auto& [u, p_old] : nb_probs) {
        auto cnt_it = elite_nb.find(u);
        const double p_elite = cnt_it != elite_nb.end() ? cnt_it->second : 0.0;
        p_old = (1.0 - LEARNING_RATE) * p_old + LEARNING_RATE * p_elite;
      }
    }
  }
}

// CEM-based solve
// ---------------------------------------------------------------------------

Solution CEMPlanner::solve(std::string& additional_info)
{
  set_Scatter();
  g_status_lines = 0;
  g_probe_agent  = 0;
  const auto solve_start = std::chrono::steady_clock::now();

  // 1. Build initial solution from random rollouts.
  std::vector<RolloutResult> global_elite;
  global_elite.reserve(CEM_ELITE_COUNT);
  auto initial_nsp = create_initial_policy(ins->N, global_elite, 2000, 100);

  std::cout << "Best init cost: " << global_elite[0].cost << std::endl;
  Solution current_solution;
  for (auto& cfg : global_elite[0].configs) current_solution.push_back(cfg);

  // Helper: compute sum-of-costs of a solution segment against given goals.
  auto compute_soc = [&](const Solution& sol, const Config& goals) -> uint {
    uint cost = 0;
    const int nag = static_cast<int>(ins->N);
    for (size_t step = 0; step + 1 < sol.size(); ++step) {
      for (int a = 0; a < nag; ++a) {
        if (sol[step][a] != goals[a] || sol[step + 1][a] != goals[a])
          cost += 1;
      }
    }
    return cost;
  };

  // Refiner interval widths cycling: 0.15 → 0.25 → 0.33 → repeat.
  static const double WIDTHS[]  = {0.15, 0.25, 0.33};
  static const int    NUM_W     = 3;

  uint outer_gen    = 0;
  bool outer_stopped = false;

  while (!is_expired(deadline) && !outer_stopped) {
    const double width       = WIDTHS[outer_gen % NUM_W];
    const int    total_steps = static_cast<int>(current_solution.size());
    if (total_steps < 2) break;

    // Number of steps in each refiner interval (at least 1).
    const int interval_steps = std::max(3, static_cast<int>(width * total_steps));

    // ── Slide the refiner interval across the full solution ─────────────────
    int mid1_idx = 0;
    while (!outer_stopped && !is_expired(deadline)) {
      const int cur_total = static_cast<int>(current_solution.size());
      if (mid1_idx >= cur_total - 1) break;

      const int mid2_idx = std::min(mid1_idx + interval_steps, cur_total - 1);

      const Config& mid1_config = current_solution[mid1_idx];
      const Config& mid2_config = current_solution[mid2_idx];

      // The three parts (part1 and part3 are kept fixed).
      Solution part1 = mid1_idx == 0 ? Solution() : Solution(current_solution.begin(),
                    current_solution.begin() + mid1_idx);

      Solution interval(current_solution.begin() + mid1_idx,
                        current_solution.begin() + mid2_idx);

      Solution part3(current_solution.begin() + mid2_idx,
                    current_solution.end());
      
      // SoC of the interval against its own sub-goal (what CEM minimises).
      const uint soc_interval_before = compute_soc(interval, ins->goals);

      // Complementary cost: parts not being optimised, measured against global goal.
      const uint soc_complement = compute_soc(part1, ins->goals)
                                + compute_soc(part3, ins->goals);

      // Sub-problem: mid1_config → mid2_config, borrowing ins->G.
      Instance sub_ins(*ins, mid1_config, mid2_config);

      OuterContext ctx;
      ctx.gen               = outer_gen;
      ctx.start_time        = solve_start;
      ctx.complementary_soc = soc_complement;

      CEMPlanner cem_ref(&sub_ins, deadline, MT, verbose, objective, RESTART_RATE);
      g_status_lines = 0;
      Solution new_interval =
          cem_ref.solve_with_cem(additional_info, ctx, &outer_stopped);

      // Replace the interval only if the refined SoC improved.
      int next_mid1 = mid2_idx;
      if (!new_interval.empty()) {
        const uint soc_interval_after = compute_soc(new_interval, ins->goals);
        if (soc_interval_after < soc_interval_before) {
          Solution new_sol;
          // The total size is just the sum of the parts
          new_sol.reserve(part1.size() + new_interval.size() + part3.size());

          // Append them sequentially without any offsets
          new_sol.insert(new_sol.end(), part1.begin(), part1.end());
          new_sol.insert(new_sol.end(), new_interval.begin(), new_interval.end());
          new_sol.insert(new_sol.end(), part3.begin(), part3.end());

          current_solution = std::move(new_sol);

          // Next interval starts right after the new interval.
          next_mid1 = mid1_idx + static_cast<int>(new_interval.size()) - 1;
        }
      }

      mid1_idx = next_mid1;
    }

    ++outer_gen;
  }

  return current_solution;
}

Solution CEMPlanner::solve_with_cem(std::string& additional_info,
                                    const OuterContext& outer,
                                    bool* outer_stop)
{
  set_Scatter();
  g_status_lines = 0;
  g_probe_agent  = 0;
  const auto solve_start = std::chrono::steady_clock::now();

  // 1. Build initial policy from random rollouts.
  std::vector<RolloutResult> global_elite;
  global_elite.reserve(CEM_ELITE_COUNT);
  auto initial_nsp = create_initial_policy(ins->N, global_elite, 2000, 100);

  std::cout << "Best init cost: " << global_elite[0].cost << std::endl;
  global_elite.clear();

  // 2. Translate to a ProbabilityPolicy.
  PolicyRandomizer randomizer(&ins->G, MT, PRS_NUM_THREADS);
  ProbabilityPolicy prob_policy(ins->N);
  const auto& agent_pols = initial_nsp.get_policies();
  for (uint a = 0; a < static_cast<uint>(ins->N); ++a) {
    if (a < agent_pols.size())
      prob_policy[a] = to_probability_policy(agent_pols[a], INIT_LAPLACE_SMOOTHING);
  }

  uint best_cost = UINT_MAX;
  std::vector<Config> best_configs;

  std::vector<std::mt19937> thread_rngs(PRS_NUM_THREADS);
  if (MT != nullptr)
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) thread_rngs[i].seed((*MT)());
  else
    for (uint i = 0; i < PRS_NUM_THREADS; ++i) thread_rngs[i].seed(i + 1);

  uint gen = 0;
  while (!is_expired(deadline)) {
    ++gen;
    auto key = read_key_event_nonblocking();
    if (key.event == KeyEvent::StopInner || key.event == KeyEvent::StopOuter) {
      // Finalise the status display before printing the stop message.
      if (g_status_lines > 0) std::cout << "\033[" << g_status_lines << "A\033[J";
      g_status_lines = 0;
      std::cout << "CEMPlanner: interrupted by user ("
                << (key.event == KeyEvent::StopInner ? "Space" : "Escape")
                << ").\n" << std::flush;
      if (key.event == KeyEvent::StopOuter && outer_stop != nullptr)
        *outer_stop = true;
      break;
    }
    if (key.event == KeyEvent::ShiftUp    || key.event == KeyEvent::ShiftDown ||
        key.event == KeyEvent::ShiftLeft  || key.event == KeyEvent::ShiftRight) {
      cem_ui_handle_scroll(key.event);
    }
    if (key.event == KeyEvent::Char) {
      if (key.ch == 'a' || key.ch == 'A') {
        prompt_agent_id(ins->N);
      }
      if (key.ch == 'l' || key.ch == 'L') {
        char sub = read_key_blocking();
        if (sub == 'p' || sub == 'P') {
          prompt_gen_laplace();
        } else if (sub == 'r' || sub == 'R') {
          if (prompt_base_lr()) {
            gen = 0;
          }
        }
      }
    }

    LEARNING_RATE = LEARNING_RATE_FUNC(gen, BASE_LEARNING_RATE);

    // 3-4. Generate and evaluate candidates.
    // const uint elite_lowest = global_elite.empty() ? UINT_MAX : global_elite.back().cost;
    auto eval_results = run_candidate_rollouts(
        prob_policy, randomizer, thread_rngs, CEM_NUM_CANDIDATES);

    // 5. Select elite.
    select_elite(eval_results, CEM_ELITE_COUNT, best_cost, best_configs);
    if (eval_results.empty()) continue;

    // Update global_elite: replace worse entries with better new ones.
    int new_elite_count = update_global_elite(global_elite, eval_results);

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();
    draw_cem_status(gen, elapsed,
                    static_cast<int>(eval_results.size()), new_elite_count,
                    best_cost, prob_policy, ins->N, ins, outer);

    // 6. Update probability policy from elite rollout moves.
    if (new_elite_count > 0)
      update_policy_with_elite(prob_policy, global_elite);

  }

  if (best_configs.empty()) return Solution{};

  // run_stall_test(prob_policy);

  Solution solution;
  // solution.push_back(ins->starts);
  for (auto& cfg : best_configs) solution.push_back(cfg);
  return solution;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Model-entropy diagnostic
// ---------------------------------------------------------------------------

void CEMPlanner::print_model_entropy(const ProbabilityPolicy& prob_policy, uint gen) const
{
  static const std::vector<double> THRESHOLDS = {0.8, 0.9, 0.95, 0.99};
  // Pick representative agent indices; skip those out of range.
  std::vector<int> probe_agents = {0};

  for (int a : probe_agents) {
    if (a >= ins->N) continue;
    const auto& agent_pol = prob_policy[a];
    const uint visited = static_cast<uint>(agent_pol.vertex_probs.size());
    if (visited == 0) {
      continue;
    }

    std::cout << "  [entropy] gen=" << gen << " agent=" << a
              << " visited=" << visited;
    for (double thresh : THRESHOLDS) {
      uint confident = 0;
      for (const auto& [v, nb_probs] : agent_pol.vertex_probs) {
        double best = 0.0;
        for (const auto& [u, p] : nb_probs)
          if (p > best) best = p;
        if (best > thresh) ++confident;
      }
      std::cout << " p>" << thresh << "=" << std::fixed << std::setprecision(2)
                << (100.0 * confident / visited) << "%";
    }
    std::cout << std::endl;
  }
}

// PIBT vs PolicyPIBT single-step speed benchmark
// ---------------------------------------------------------------------------

void CEMPlanner::test_pibt_speed()
{
  constexpr uint BENCH_N = 50000;
  std::mt19937 bench_rng(42);
  PIBT bench_pibt(ins, D, &bench_rng);

  // Build a dummy ProbabilityPolicy (empty — randomizer will fall back to uniform).
  PolicyRandomizer bench_randomizer(&ins->G, &bench_rng, 1);
  ProbabilityPolicy dummy_prob_policy(ins->N);
  std::vector<std::mt19937> single_rng(1); single_rng[0].seed(42);
  auto bench_policies = bench_randomizer(dummy_prob_policy, ins, single_rng, 1);
  auto bench_pol = bench_policies[0];
  PolicyPIBT bench_pp(ins, D, MT, bench_pol, scatter);

  auto* H_step = new HNode(ins->starts, D, nullptr, 0, 0);
  LNode unconstrained;
  uint pibt_ok = 0, pp_ok = 0;

  auto t0 = std::chrono::steady_clock::now();
  for (uint i = 0; i < BENCH_N; ++i) {
    Config C_new(ins->N, nullptr);
    if (bench_pibt.get_new_config(H_step, &unconstrained, C_new)) ++pibt_ok;
  }
  auto t1 = std::chrono::steady_clock::now();
  double pibt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[bench] PIBT       step: " << BENCH_N << " steps in "
            << std::fixed << std::setprecision(1) << pibt_ms << " ms"
            << "  (" << std::setprecision(3) << pibt_ms / BENCH_N << " ms/step)"
            << "  ok=" << pibt_ok << "\n";

  auto t2 = std::chrono::steady_clock::now();
  for (uint i = 0; i < BENCH_N; ++i) {
    Config C_new(ins->N, nullptr);
    if (bench_pp.get_new_config(H_step, &unconstrained, C_new)) ++pp_ok;
  }
  auto t3 = std::chrono::steady_clock::now();
  double pp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  std::cout << "[bench] PolicyPIBT step: " << BENCH_N << " steps in "
            << std::fixed << std::setprecision(1) << pp_ms << " ms"
            << "  (" << std::setprecision(3) << pp_ms / BENCH_N << " ms/step)"
            << "  ok=" << pp_ok << "\n";

  std::cout << "[bench] PolicyPIBT/PIBT slowdown: "
            << std::setprecision(2) << pp_ms / pibt_ms << "x\n\n";
  delete H_step;
}

// Interactive stall-simulation test
// ---------------------------------------------------------------------------

void CEMPlanner::run_stall_test(const ProbabilityPolicy& prob_policy)
{
  std::cout << "\n=== Stall-simulation test (q / empty Enter to quit) ===" << std::endl;

  PolicyRandomizer randomizer(&ins->G, MT, PRS_NUM_THREADS);
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
    std::vector<std::mt19937> single_rng(1);
    if (MT) single_rng[0].seed((*MT)());
    auto step_policies = randomizer(prob_policy, ins, single_rng, 1);
    auto ce_pol = step_policies[0];

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
    PolicyPIBT pp_step(ins, D, MT, ce_pol, scatter);
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

    // Generate all stall policies before spawning threads.
    auto stall_policies = randomizer(prob_policy, ins, stall_rngs, STALL_NUM_POLICIES);

    struct StallResult { RolloutResult res; };
    std::vector<std::future<StallResult>> stall_futures;
    stall_futures.reserve(STALL_NUM_POLICIES);
    for (uint i = 0; i < STALL_NUM_POLICIES; ++i) {
      auto pol = stall_policies[i];
      stall_futures.push_back(std::async(std::launch::async,
          [this, pol, &C_stalled]() -> StallResult {
            PolicyPIBT pp(ins, D, MT, pol, scatter);
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
    // sol.push_back(ins->starts);
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

void CEMPlanner::set_Scatter()
{
  info(1, verbose, deadline, "start computing SUO");
  auto scatter_deadline =
      Deadline(deadline == nullptr
                   ? INT_MAX
                   : (deadline->time_limit_ms - elapsed_ms(deadline)) / 2);
  int SCATTER_MARGIN = 10;                 
  scatter = new Scatter(ins, &D, &scatter_deadline, 3, verbose - 4, SCATTER_MARGIN);

  scatter->construct();
  info(1, verbose, deadline, "finish computing SUO",
       ", collision count: ", scatter->CT.collision_cnt,
       ", scatter margin: ", scatter->cost_margin,
       ", sum_of_path_length: ", scatter->sum_of_path_length);
}

