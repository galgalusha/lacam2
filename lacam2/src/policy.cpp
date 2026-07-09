#include "../include/policy.hpp"
#include "../include/planner.hpp"

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


/*
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
*/
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

/*
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
*/

// ---------------------------------------------------------------------------
// CEM helpers
// ---------------------------------------------------------------------------

AgentProbabilityPolicy to_probability_policy(const AgentScores& ap, double laplace_factor)
{
  AgentProbabilityPolicy result;
  result.vertex_probs.reserve(ap.vertex_scores.size());

  for (const auto& [from, ns] : ap.vertex_scores) {
    if (ns.scores.empty()) continue;

    // Full candidate set: neighbors + stay-in-place.
    std::vector<Vertex*> candidates = from->neighbor;
    candidates.push_back(from);
    const double num_candidates = static_cast<double>(candidates.size());

    double total = 0.0;
    for (const auto& [nb, score] : ns.scores) total += score;

    auto& dist = result.vertex_probs[from];
    if (laplace_factor <= 0.0) {
      // Original behaviour: only visited neighbors get non-zero probability.
      for (const auto& [nb, score] : ns.scores) dist[nb] = score / total;
    } else {
      // Laplace smoothing: P(u|v) = (count(u) + α) / (total + α * |candidates|)
      const double denom = total + laplace_factor * num_candidates;
      for (Vertex* u : candidates) {
        auto it = ns.scores.find(u);
        const double count = (it != ns.scores.end()) ? static_cast<double>(it->second) : 0.0;
        dist[u] = (count + laplace_factor) / denom;
      }
    }
  }
  return result;
}

AgentDeterministicPolicy PolicyRandomizer::sample_agent_policy(
    const AgentProbabilityPolicy& prob_policy) const
{
  const Graph& G = *graph;
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

  std::vector<std::shared_ptr<DeterministicPolicy>> policies;
  policies.reserve(num_policies);
  PolicyRandomizer local_copy = *this;
  local_copy.graph = &ins->G;
  for (uint p = 0; p < num_policies; ++p) {
    local_copy.rng = &rngs[p % rngs.size()];
    std::vector<AgentDeterministicPolicy> disc(N);
    for (uint a = 0; a < N; ++a)
      disc[a] = local_copy.sample_agent_policy(prob_policy[a]);
    policies.push_back(
        std::make_shared<DeterministicPolicy>(std::move(disc), ins, local_copy.rng));
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
}

void DeterministicPolicy::set_tie_breakers(uint agent_id, Vertex* current,
                                            const Vertices& neighbors,
                                            std::vector<float>& tie_breakers_by_id)
{
  // Zero out all neighbor entries first.
  for (Vertex* v : neighbors) tie_breakers_by_id[v->id] = 0.0f;

  if (agent_id >= discrete.size()) return;
  const auto& deterministic = discrete[agent_id];

  auto it = deterministic.rankings.find(current);
  if (it != deterministic.rankings.end()) {
    for (Vertex* v : neighbors) {
      auto sit = it->second.find(v);
      if (sit != it->second.end()) tie_breakers_by_id[v->id] = sit->second;
    }
  } else {
    // Blind spot: assign random scores so PolicyPIBT gets varied tie-breaking.
    for (Vertex* v : neighbors) tie_breakers_by_id[v->id] = get_random_float(rng, -0.9f, 0.0f);
  }
}
