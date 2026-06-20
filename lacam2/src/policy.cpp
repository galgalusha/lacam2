#include "../include/policy.hpp"

Vertex* RandomPolicy::get_preferred_neighbor(const Config& /*C*/,
                                              uint /*agent_id*/,
                                              const Vertices& candidates)
{
  if (candidates.empty()) return nullptr;
  if (candidates.size() == 1) return candidates[0];
  std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
  return candidates[dist(*MT)];
}
