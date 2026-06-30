#include "../include/policy.hpp"

#include <algorithm>
#include <numeric>

void AgentPolicy::record_move(Vertex* from, Vertex* to)
{
  vertex_scores[from].scores[to]++;
}

void AgentPolicy::finish_recording(const Instance* ins, std::mt19937* MT)
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

void AgentPolicy::randomize_scores(std::mt19937* rng)
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

std::unordered_map<Vertex*, float> NeighborScorePolicy::get_neighbor_scores(
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

void NeighborScorePolicy::finish_recording(const Instance* ins)
{
  for (auto& ap : policies) {
    ap.finish_recording(ins, MT);
  }
}

void NeighborScorePolicy::randomize_agent_blind_scores(uint agent_idx,
                                                        const Instance* ins,
                                                        std::mt19937* rng)
{
  if (agent_idx < policies.size()) {
    policies[agent_idx].finish_recording(ins, rng);
  }
}

void NeighborScorePolicy::randomize_agent_scores(uint agent_idx,
                                                  std::mt19937* rng)
{
  if (agent_idx < policies.size()) {
    policies[agent_idx].randomize_scores(rng);
  }
}

void NeighborScorePolicy::randomize_all_scores(std::mt19937* rng)
{
  for (auto& ap : policies) {
    ap.randomize_scores(rng);
  }
}
