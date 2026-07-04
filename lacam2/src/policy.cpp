#include "../include/policy.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <queue>
#include <queue>

void AgentScores::record_move(Vertex* from, Vertex* to)
{
  vertex_scores[from].scores[to]++;
}

void AgentScores::finish_recording(const Instance* ins, std::mt19937* MT)
{
  std::uniform_real_distribution<float> rand_dist(-0.9f, 0.0f);
  for (Vertex* v : ins->G.V) {
    if (v == nullptr) continue;
    Vertices candidates = v->neighbor;
    candidates.push_back(v);
    blind_score_map[v].reserve(candidates.size());
    for (Vertex* nb : candidates) {
      blind_score_map[v][nb] = rand_dist(*MT);
    }
  }
}

void AgentScores::randomize_scores(std::mt19937* rng)
{
  for (auto& [from, ns] : vertex_scores) {
    auto& scores = ns.scores;
    if (scores.empty()) continue;
    // Collect neighbor keys and assign a shuffled permutation of [1..n].
    std::vector<Vertex*> keys;
    keys.reserve(scores.size());
    for (auto& [v, _] : scores) keys.push_back(v);
    std::vector<int> vals(keys.size());
    std::iota(vals.begin(), vals.end(), 1);  // [1, 2, ..., n]
    std::shuffle(vals.begin(), vals.end(), *rng);
    for (size_t i = 0; i < keys.size(); ++i) scores[keys[i]] = vals[i];
  }
}

std::unordered_map<Vertex*, float> RandomPolicy::get_neighbor_scores(
    const Config& /*C*/, uint /*agent_id*/, const Vertices& neighbors)
{
  std::unordered_map<Vertex*, float> result;
  result.reserve(neighbors.size());
  std::uniform_real_distribution<float> dist(-0.9f, 0.0f);
  for (Vertex* v : neighbors) result[v] = dist(*MT);
  return result;
}

std::unordered_map<Vertex*, float> ScorePolicy::get_neighbor_scores(
    const Config& C, uint agent_id, const Vertices& neighbors)
{
  std::unordered_map<Vertex*, float> result;
  result.reserve(neighbors.size());

  auto blind_result = [&](Vertex* current) {
    const auto bit = policies[agent_id].blind_score_map.find(current);
    for (Vertex* v : neighbors) {
      float score = 0.0f;
      if (bit != policies[agent_id].blind_score_map.end()) {
        const auto& nm = bit->second;
        auto nit = nm.find(v);
        if (nit != nm.end()) score = nit->second;
      }
      result[v] = score;
    }
    ++random_count;
    return result;
  };

  if (agent_id >= policies.size()) {
    // No policy at all for this agent — scores stay 0.
    for (Vertex* v : neighbors) result[v] = 0.0f;
    ++random_count;
    return result;
  }

  Vertex* current = C[agent_id];
  const auto& ap = policies[agent_id];
  auto vit = ap.vertex_scores.find(current);
  if (vit == ap.vertex_scores.end()) return blind_result(current);

  const auto& score_map = vit->second.scores;

  // Find max score among neighbors that have an entry.
  int max_score = 0;
  bool any = false;
  for (Vertex* v : neighbors) {
    auto sit = score_map.find(v);
    if (sit != score_map.end()) {
      if (!any || sit->second > max_score) {
        max_score = sit->second;
        any = true;
      }
    }
  }

  if (!any || max_score <= 0) return blind_result(current);

  ++deterministic_count;
  // Map scores to [-0.9, 0]: score/max_score * -0.9; unscored neighbors get 0
  for (Vertex* v : neighbors) {
    auto sit = score_map.find(v);
    result[v] = (sit != score_map.end() && sit->second > 0)
                    ? (static_cast<float>(sit->second) / max_score) * -0.9f
                    : 0.0f;
  }
  return result;
}

void ScorePolicy::finish_recording(const Instance* ins)
{
  for (auto& ap : policies) {
    ap.finish_recording(ins, MT);
  }
}

void ScorePolicy::randomize_agent_blind_scores(uint agent_idx,
                                                        const Instance* ins,
                                                        std::mt19937* rng)
{
  if (agent_idx < policies.size()) {
    policies[agent_idx].finish_recording(ins, rng);
  }
}

void ScorePolicy::randomize_agent_scores(uint agent_idx,
                                                  std::mt19937* rng)
{
  if (agent_idx < policies.size()) {
    policies[agent_idx].randomize_scores(rng);
  }
}

void ScorePolicy::randomize_all_scores(std::mt19937* rng)
{
  for (auto& ap : policies) {
    ap.randomize_scores(rng);
  }
}

// ---------------------------------------------------------------------------
// ProbabilityPolicy
// ---------------------------------------------------------------------------

std::unordered_map<Vertex*, float> ProbabilityPolicy::get_neighbor_scores(
    const Config& C, uint agent_id, const Vertices& neighbors)
{
  std::unordered_map<Vertex*, float> result;
  result.reserve(neighbors.size());
  for (Vertex* v : neighbors) result[v] = 0.0f;

  if (agent_id >= probs.size()) return result;
  Vertex* current = C[agent_id];
  const auto& nb_probs_map = probs[agent_id].vertex_probs;

  auto vit = nb_probs_map.find(current);

  if (vit == nb_probs_map.end() || vit->second.empty()) {
    // Blind spot: sample uniformly from neighbors.
    std::uniform_int_distribution<size_t> udist(0, neighbors.size() - 1);
    result[neighbors[udist(*rng)]] = -0.9f;
    return result;
  }

  const auto& nb_probs = vit->second;
  std::uniform_real_distribution<double> udist(0.0, 1.0);
  double r = udist(*rng);
  double cumul = 0.0;
  Vertex* chosen = nb_probs.begin()->first;
  for (const auto& [nb, p] : nb_probs) {
    cumul += p;
    chosen = nb;
    if (r < cumul) break;
  }
  result[chosen] = -0.9f;
  return result;
}

// ---------------------------------------------------------------------------
// CEM helpers
// ---------------------------------------------------------------------------

AgentProbabilityPolicy to_probability_policy(const AgentScores& ap)
{
  AgentProbabilityPolicy result;
  result.vertex_probs.reserve(ap.vertex_scores.size());
  result.priority_dist.reserve(ap.priority_records.size());
  
  for (const auto& [from, ns] : ap.vertex_scores) {
    if (ns.scores.empty()) continue;
    double total = 0.0;
    for (const auto& [nb, score] : ns.scores) total += score;
    // total > 0: scores are positive integers from record_move increments.
    auto& dist = result.vertex_probs[from];
    for (const auto& [nb, score] : ns.scores) dist[nb] = score / total;
  }

  // Compute Gaussian priority distribution parameters from priority_records.
  for (const auto& [from, records] : ap.priority_records) {
    const size_t K = records.size();
    if (K == 0) continue;

    // Sample mean.
    double mu = 0.0;
    for (uint r : records) mu += r;
    mu /= static_cast<double>(K);

    // Sample standard deviation (population formula).
    double variance = 0.0;
    for (uint r : records) {
      double diff = static_cast<double>(r) - mu;
      variance += diff * diff;
    }
    double sigma = std::sqrt(variance / static_cast<double>(K));

    // Safety floor: prevent sigma == 0 from killing exploration.
    if (sigma < 1.0) sigma = 1.0;

    result.priority_dist[from] = {mu, sigma};
  }

  return result;
}

// Multi-source BFS: propagates PriorityDist from known vertices to all
// reachable vertices (nearest-neighbour copy). Run once per agent in
// PolicyRandomizer::operator(); each per-policy sample then just draws
// from the returned grid. Wall cells stay at the default {0.0, 1.0}.
static std::vector<PriorityDist> floodfill_priority_dist(
    const Graph& G,
    const std::unordered_map<Vertex*, PriorityDist>& known)
{
  const size_t grid_size = static_cast<size_t>(G.width) * G.height;
  std::vector<PriorityDist> grid(grid_size, {0.0, 1.0});
  std::vector<bool> visited(grid_size, false);

  std::queue<Vertex*> q;
  for (const auto& [v, dist] : known) {
    grid[v->index] = dist;
    visited[v->index] = true;
    q.push(v);
  }

  while (!q.empty()) {
    Vertex* cur = q.front(); q.pop();
    for (Vertex* nb : cur->neighbor) {
      if (nb == nullptr) continue;
      if (visited[nb->index]) continue;
      grid[nb->index] = grid[cur->index];
      visited[nb->index] = true;
      q.push(nb);
    }
  }

  return grid;
}

static AgentDeterministicPolicy sample_agent_policy(
    const AgentProbabilityPolicy& prob_policy,
    const std::vector<PriorityDist>& filled_dist,
    const Graph& G,
    std::mt19937* rng)
{
  AgentDeterministicPolicy result;
  std::uniform_real_distribution<double> uniform;

  for (const auto& [v, nb_probs] : prob_policy.vertex_probs) {
    if (nb_probs.empty()) continue;

    // Weighted random sampling without replacement to build full strict ranking.
    std::vector<std::pair<Vertex*, double>> remaining(nb_probs.begin(), nb_probs.end());
    float score = -0.9f;

    while (!remaining.empty()) {
      double total = 0.0;
      for (const auto& [nb, p] : remaining) total += p;

      size_t chosen_idx = 0;

      // Safety check: prevent UB in uniform_real_distribution if total is 0.
      if (total > 1e-9) {
        double r = uniform(*rng, std::uniform_real_distribution<double>::param_type(0.0, total));
        double cumul = 0.0;
        for (size_t i = 0; i < remaining.size(); ++i) {
          cumul += remaining[i].second;
          chosen_idx = i;
          if (r <= cumul) break;
        }
      }

      result.rankings[v][remaining[chosen_idx].first] = score;
      score += 0.1f;
      remaining.erase(remaining.begin() + chosen_idx);
    }
  }

  // Sample priority score for every passable vertex from the pre-filled dist grid.
  const size_t grid_size = static_cast<size_t>(G.width) * G.height;
  result.priority_grid.resize(grid_size, 0.0);
  for (Vertex* v : G.V) {
    std::normal_distribution<double> nd(filled_dist[v->index].mu,
                                        filled_dist[v->index].sigma);
    result.priority_grid[v->index] = nd(*rng);
  }

  return result;
}

std::vector<std::shared_ptr<DeterministicPolicy>> PolicyRandomizer::operator()(
    const ProbabilityPolicy& prob_policy,
    const Instance* ins,
    std::vector<std::mt19937>& rngs,
    uint num_policies) const
{
  const Graph& G = ins->G;
  const uint N = static_cast<uint>(prob_policy.size());

  // Pre-compute floodfilled PriorityDist grid once per agent.
  std::vector<std::vector<PriorityDist>> filled_dists(N);
  for (uint a = 0; a < N; ++a)
    filled_dists[a] = floodfill_priority_dist(G, prob_policy[a].priority_dist);

  std::vector<std::shared_ptr<DeterministicPolicy>> policies;
  policies.reserve(num_policies);

  for (uint p = 0; p < num_policies; ++p) {
    std::mt19937* rng = &rngs[p % rngs.size()];
    std::vector<AgentDeterministicPolicy> disc(N);
    for (uint a = 0; a < N; ++a)
      disc[a] = sample_agent_policy(prob_policy[a], filled_dists[a], G, rng);
    policies.push_back(
        std::make_shared<DeterministicPolicy>(std::move(disc), ins, rng));
  }

  return policies;
}

void test_randomizer()
{
  // Build a 4x4 fully-open grid (all '.' = passable).
  const std::vector<std::string> grid = {
    "....",
    "....",
    "....",
    "...."
  };
  Graph G(grid);  // width=4, height=4; vertex index = y*4+x

  // Locate (0,0) and (3,3) by their index.
  Vertex* v00 = G.U[0 * 4 + 0];  // index 0
  Vertex* v33 = G.U[3 * 4 + 3];  // index 15

  // Set up one agent with known priority distributions at two vertices.
  AgentProbabilityPolicy app;
  app.priority_dist[v00] = {1.0, 0.1};   // (0,0): mu=1, low noise
  app.priority_dist[v33] = {5.0, 0.1};   // (3,3): mu=5, low noise

  // Floodfill once.
  std::vector<PriorityDist> filled = floodfill_priority_dist(G, app.priority_dist);

  // Sample one deterministic policy from the filled dist.
  std::mt19937 rng(42);
  AgentDeterministicPolicy det = sample_agent_policy(app, filled, G, &rng);

  // Print the 2D grid of sampled priority values.
  std::cout << "Priority grid (4x4), sampled from floodfilled dist:\n";
  for (uint y = 0; y < G.height; ++y) {
    for (uint x = 0; x < G.width; ++x) {
      uint idx = y * G.width + x;
      std::cout << std::fixed;
      std::cout.precision(2);
      std::cout << det.priority_grid[idx];
      if (x + 1 < G.width) std::cout << "  ";
    }
    std::cout << '\n';
  }
}

std::unordered_map<Vertex*, float> DeterministicPolicy::get_neighbor_scores(
    const Config& C, uint agent_id, const Vertices& neighbors)
{
  std::unordered_map<Vertex*, float> result;
  result.reserve(neighbors.size());
  for (Vertex* v : neighbors) result[v] = 0.0f;

  if (agent_id >= discrete.size()) return result;
  Vertex* current = C[agent_id];
  const auto& deterministic = discrete[agent_id];

  auto it = deterministic.rankings.find(current);
  if (it != deterministic.rankings.end()) {
    for (Vertex* v : neighbors) {
      auto sit = it->second.find(v);
      result[v] = (sit != it->second.end()) ? sit->second : 0.0f;
    }
  } else {
    // Blind spot: assign random scores so PolicyPIBT gets varied tie-breaking.
    for (Vertex* v : neighbors) result[v] = get_random_float(rng, -0.9f, 0.0f);
  }
  return result;
}
