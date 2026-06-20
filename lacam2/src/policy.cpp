#include "../include/policy.hpp"

#include <algorithm>
#include <numeric>

void AgentPolicy::record_move(Vertex* from, Vertex* to)
{
  vertex_scores[from].scores[to]++;
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

  std::uniform_real_distribution<float> rand_dist(-0.9f, 0.0f);
  auto random_result = [&]() {
    for (Vertex* v : neighbors) result[v] = rand_dist(*MT);
    ++random_count;
    return result;
  };

  if (agent_id >= policies.size()) return random_result();

  Vertex* current = C[agent_id];
  const auto& ap = policies[agent_id];
  auto vit = ap.vertex_scores.find(current);
  if (vit == ap.vertex_scores.end()) return random_result();

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

  if (!any || max_score <= 0) return random_result();

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
