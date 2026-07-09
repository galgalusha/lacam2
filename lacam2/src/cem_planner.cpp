#include "../include/cem_planner.hpp"
#include "../include/pibt.hpp"
#include "../include/post_processing.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// Returns the key pressed (or 0 if none). Non-blocking.
static char read_key_nonblocking()
{
  struct termios oldt, newt;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);

  int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, old_flags | O_NONBLOCK);

  char c = 0;
  read(STDIN_FILENO, &c, 1);

  fcntl(STDIN_FILENO, F_SETFL, old_flags);
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return c;
}

// Hyper Parameters

static uint CEM_NUM_CANDIDATES        = 100;
static int CEM_ELITE_COUNT            = 10;
static double INIT_LAPLACE_SMOOTHING  = 2.0;
static double GEN_LAPLACE_SMOOTHING   = 0.02;
static double BASE_LEARNING_RATE      = 0.2;
static auto LEARNING_RATE_FUNC   = [](int gen, float base) 
                                   { return base * sqrt(100.0 / (100.0 + gen)); };
static float LEARNING_RATE            = LEARNING_RATE_FUNC(0, BASE_LEARNING_RATE);

// ---- Live status display -----------------------------------------------

static uint g_status_lines = 0;  // how many lines the last status block occupied
static int  g_probe_agent  = 0;  // agent whose entropy stats are shown

static void draw_cem_status(uint gen, double elapsed_sec,
                             int elite_size, int new_global_elite, uint best_cost,
                             const ProbabilityPolicy& prob_policy, int num_agents)
{
  if (g_status_lines > 0)
    std::cout << "\033[" << g_status_lines << "A\033[J";

  const int mins = static_cast<int>(elapsed_sec / 60.0);
  const double secs = elapsed_sec - mins * 60.0;

  // ── Box geometry ────────────────────────────────────────────────
  // Left panel inner width (LW), right panel inner width (RW).
  // Every line is exactly LW+RW+7 visible chars wide.
  constexpr int LW = 24;
  constexpr int RW = 39;
  constexpr int FW = LW + RW + 3;  // full inner width = 66

  // Repeat a UTF-8 border glyph n times.
  auto rep = [](const char* g, int n) -> std::string {
    std::string s; for (int i = 0; i < n; ++i) s += g; return s;
  };
  // Pad/truncate a plain-ASCII string to exactly w chars.
  auto pad = [](std::string s, int w) -> std::string {
    if ((int)s.size() > w) s.resize(w);
    else s.append(w - (int)s.size(), ' ');
    return s;
  };

  // Border lines (all LW+RW+7 wide).
  const std::string full_top  = "\u2554" + rep("\u2550", FW+2)        + "\u2557";
  const std::string split_sep = "\u2560" + rep("\u2550", LW+2) + "\u2566" + rep("\u2550", RW+2) + "\u2563";
  const std::string bot       = "\u255a" + rep("\u2550", LW+2) + "\u2569" + rep("\u2550", RW+2) + "\u255d";

  // Two-column content row (pure-ASCII content only — no UTF-8 in L or R).
  auto row = [&](const std::string& L, const std::string& R) -> std::string {
    return "\u2551 " + pad(L, LW) + " \u2551 " + pad(R, RW) + " \u2551";
  };
  // Right-panel internal horizontal separator; left panel stays open.
  // L must be pure ASCII (pad() counts bytes).
  auto rsep = [&](const std::string& L) -> std::string {
    return "\u2551 " + pad(L, LW) + " \u2560" + rep("\u2550", RW+2) + "\u2563";
  };

  // ── Entropy data ─────────────────────────────────────────────────
  static constexpr double THRESH[4] = {0.80, 0.90, 0.95, 0.99};
  double cpct[4] = {};
  uint visited = 0;
  if (g_probe_agent < num_agents) {
    const auto& apol = prob_policy[g_probe_agent];
    visited = static_cast<uint>(apol.vertex_probs.size());
    if (visited > 0) {
      for (int ti = 0; ti < 4; ++ti) {
        uint cnt = 0;
        for (const auto& [v, nb] : apol.vertex_probs) {
          double best = 0.0;
          for (const auto& [u, p] : nb) if (p > best) best = p;
          if (best > THRESH[ti]) ++cnt;
        }
        cpct[ti] = 100.0 * cnt / visited;
      }
    }
  }

  // ── Format helpers (pure ASCII output) ───────────────────────────
  auto fi = [](long long v, int w) -> std::string {
    std::ostringstream s; s << std::setw(w) << v; return s.str();
  };
  auto fd = [](double v, int w, int p) -> std::string {
    std::ostringstream s;
    s << std::fixed << std::setprecision(p) << std::setw(w) << v;
    return s.str();
  };

  // Entropy table row (pure ASCII, fits within LW).
  auto erow = [&](int ti) -> std::string {
    std::ostringstream s;
    s << "  > " << std::fixed << std::setprecision(2) << THRESH[ti]
      << "  |  " << std::setw(6) << std::fixed << std::setprecision(2) << cpct[ti] << "%";
    return s.str();  // ~20 chars, LW=24
  };

  // ── Header: GEN (bold) + TIME + BEST SoC (bold), centred ─────────
  std::string gen_str = fi((long long)gen, 6);
  std::string soc_str = (best_cost == UINT_MAX) ? "      ---" : fi((long long)best_cost, 9);
  std::string time_s;
  {
    std::ostringstream t;
    t << mins << "m" << std::fixed << std::setprecision(1) << secs << "s";
    time_s = t.str();
  }
  // Compute visible (no-ANSI) length to determine centering padding.
  const std::string hdr_vis = "GEN: " + gen_str + "   TIME: " + time_s
                             + "   BEST SoC: " + soc_str;
  int vis_len  = static_cast<int>(hdr_vis.size());
  int tot_pad  = FW - vis_len;
  int lpad_n   = (tot_pad > 0) ? tot_pad / 2 : 0;
  int rpad_n   = (tot_pad > 0) ? tot_pad - lpad_n : 0;
  const std::string hdr_bold =
      std::string(lpad_n, ' ')
    + "GEN: \033[1m" + gen_str + "\033[0m"
    + "   TIME: " + time_s
    + "   BEST SoC: \033[1m" + soc_str + "\033[0m"
    + std::string(rpad_n, ' ');

  // ── Print ─────────────────────────────────────────────────────────
  uint lines = 0;
  auto pr = [&](const std::string& s) { std::cout << s << "\n"; ++lines; };

  pr(full_top);
  // Header row: ANSI bold values — output manually (pad() can't measure ANSI)
  std::cout << "\u2551 " << hdr_bold << " \u2551\n"; ++lines;
  pr(split_sep);
  pr(row("  ENTROPY  [agent =" + fi(g_probe_agent,3) + " ]",
         "  GEN STATS"));
  pr(row("  visited =" + fi((long long)visited, 8),
         "  elite_size  :" + fi(elite_size, 6)));
  pr(row("",
         "  new_elite   :" + fi(new_global_elite, 6)));
  pr(rsep("  thresh   |  confident"));
  pr(row("  " + std::string(20, '-'), "  HYPER PARAMS"));
  pr(row(erow(0), "  lr         :" + fd(LEARNING_RATE,        9, 4)));
  pr(row(erow(1), "  lr_base    :" + fd(BASE_LEARNING_RATE,   9, 4)));
  pr(row(erow(2), "  gen_smooth :" + fd(GEN_LAPLACE_SMOOTHING,9, 4)));
  pr(row(erow(3), ""));
  pr(bot);
  pr("  \033[2m[Spc/Esc]\033[0m stop  \033[2m[a]\033[0m agent  "
     "\033[2m[l+p]\033[0m smooth  \033[2m[l+r]\033[0m lr_base+reset");

  std::cout << std::flush;
  g_status_lines = lines;
}

// Read a new probe-agent id from the user (blocking, restores terminal after).
static void prompt_agent_id(int num_agents)
{
  // Restore canonical + echo mode for interactive input.
  struct termios oldt;
  tcgetattr(STDIN_FILENO, &oldt);
  struct termios newt = oldt;
  newt.c_lflag |= (ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, old_flags & ~O_NONBLOCK);

  // Print prompt below current status block (don't erase it).
  std::cout << "Enter agent id [0-" << (num_agents - 1) << "]: " << std::flush;
  std::string line;
  if (std::getline(std::cin, line)) {
    try {
      int id = std::stoi(line);
      if (id >= 0 && id < num_agents) {
        g_probe_agent = id;
        // The next draw_cem_status call will wipe the prompt line too;
        // account for it by incrementing the line counter.
        g_status_lines += 2; // prompt line + newline from getline
      }
    } catch (...) {}
  }

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, old_flags);
}

// Read a new GEN_LAPLACE_SMOOTHING value from the user (blocking).
static void prompt_gen_laplace()
{
  struct termios oldt;
  tcgetattr(STDIN_FILENO, &oldt);
  struct termios newt = oldt;
  newt.c_lflag |= (ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, old_flags & ~O_NONBLOCK);

  std::cout << "Enter new gen_laplace_smoothing (current=" << GEN_LAPLACE_SMOOTHING << "): " << std::flush;
  std::string line;
  if (std::getline(std::cin, line)) {
    try {
      double v = std::stod(line);
      if (v > 0.0) {
        GEN_LAPLACE_SMOOTHING = v;
        g_status_lines += 2;
      }
    } catch (...) {}
  }

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, old_flags);
}

// Read a new BASE_LEARNING_RATE value from the user (blocking). Returns true if reset requested.
static bool prompt_base_lr()
{
  struct termios oldt;
  tcgetattr(STDIN_FILENO, &oldt);
  struct termios newt = oldt;
  newt.c_lflag |= (ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, old_flags & ~O_NONBLOCK);

  std::cout << "Enter new base_learning_rate (current=" << BASE_LEARNING_RATE << "): " << std::flush;
  std::string line;
  bool reset = false;
  if (std::getline(std::cin, line)) {
    try {
      double v = std::stod(line);
      if (v > 0.0 && v <= 1.0) {
        BASE_LEARNING_RATE = v;
        g_status_lines += 2;
        reset = true;
      }
    } catch (...) {}
  }

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, old_flags);
  return reset;
}

// Blocking read of a single key (used for two-key sequences like 'l'+'p').
static char read_key_blocking()
{
  struct termios oldt;
  tcgetattr(STDIN_FILENO, &oldt);
  struct termios newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, old_flags & ~O_NONBLOCK);

  char c = 0;
  read(STDIN_FILENO, &c, 1);

  fcntl(STDIN_FILENO, F_SETFL, old_flags);
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return c;
}

CEMPlanner::CEMPlanner(
    const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
    const int _verbose, const Objective _objective, const float _restart_rate)
    : Planner(_ins, _deadline, _MT, _verbose, _objective, _restart_rate)
{
}

static const uint PRS_NUM_THREADS = 7;

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
  auto policies = randomizer(prob_policy, ins, thread_rngs, num_candidates);

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
            PolicyPIBT pp(ins, D, pol);
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
  g_status_lines = 0;
  g_probe_agent  = 0;
  const auto solve_start = std::chrono::steady_clock::now();

  // 1. Build initial policy from random rollouts.
  std::vector<RolloutResult> global_elite;
  global_elite.reserve(CEM_ELITE_COUNT);
  auto initial_nsp = create_initial_policy(ins->N, global_elite, 2000, 100);

  // 2. Translate to a ProbabilityPolicy.
  PolicyRandomizer randomizer(&ins->G, MT);
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
    char key = read_key_nonblocking();
    if (key == ' ' || key == 0x1B) {
      // Finalise the status display before printing the stop message.
      if (g_status_lines > 0) std::cout << "\033[" << g_status_lines << "A\033[J";
      g_status_lines = 0;
      std::cout << "CEMPlanner: interrupted by user (Space/Escape).\n" << std::flush;
      break;
    }
    if (key == 'a' || key == 'A') {
      prompt_agent_id(ins->N);
    }
    if (key == 'l' || key == 'L') {
      char sub = read_key_blocking();
      if (sub == 'p' || sub == 'P') {
        prompt_gen_laplace();
      } else if (sub == 'r' || sub == 'R') {
        if (prompt_base_lr()) {
          gen = 0;
        }
      }
    }

    LEARNING_RATE = LEARNING_RATE_FUNC(gen, BASE_LEARNING_RATE);

    // 3-4. Generate and evaluate candidates.
    const uint elite_best = global_elite.empty() ? UINT_MAX : global_elite.front().cost;
    auto eval_results = run_candidate_rollouts(
        prob_policy, randomizer, thread_rngs, CEM_NUM_CANDIDATES, elite_best);

    // 5. Select elite.
    select_elite(eval_results, CEM_ELITE_COUNT, best_cost, best_configs);
    if (eval_results.empty()) continue;

    // Update global_elite: replace worse entries with better new ones.
    int new_elite_count = update_global_elite(global_elite, eval_results);

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - solve_start).count();
    draw_cem_status(gen, elapsed,
                    static_cast<int>(eval_results.size()), new_elite_count,
                    best_cost, prob_policy, ins->N);

    // 6. Update probability policy from elite rollout moves.
    if (new_elite_count > 0)
      update_policy_with_elite(prob_policy, global_elite);

    ++gen;
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

// Interactive stall-simulation test
// ---------------------------------------------------------------------------

void CEMPlanner::run_stall_test(const ProbabilityPolicy& prob_policy)
{
  std::cout << "\n=== Stall-simulation test (q / empty Enter to quit) ===" << std::endl;

  PolicyRandomizer randomizer(&ins->G, MT);

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

    // Generate all stall policies before spawning threads.
    auto stall_policies = randomizer(prob_policy, ins, stall_rngs, STALL_NUM_POLICIES);

    struct StallResult { RolloutResult res; };
    std::vector<std::future<StallResult>> stall_futures;
    stall_futures.reserve(STALL_NUM_POLICIES);
    for (uint i = 0; i < STALL_NUM_POLICIES; ++i) {
      auto pol = stall_policies[i];
      stall_futures.push_back(std::async(std::launch::async,
          [this, pol, &C_stalled]() -> StallResult {
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
