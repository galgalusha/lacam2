#include "../include/clustered_hnode.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

ClusteredHNode::ClusteredHNode(const Config& C, DistTable& D, HNode* parent,
                               uint g, uint h)
    : HNode(C, D, parent, g, h),
//      visit_budget(static_cast<int>(500.0 / std::sqrt(static_cast<double>(depth) + 1.0))),
//      visit_budget(static_cast<int>(10000.0 / (depth + 1.0))),
      visit_budget(static_cast<int>(1000000)),
      visits_remaining(visit_budget)
{
  // HNode constructor pushed a root LNode into HNode::search_tree.
  // We do not use that tree; cluster_trees is used instead.
  // Drain the base search_tree to keep it empty.
  while (!search_tree.empty()) search_tree.pop();
}

std::vector<std::shared_ptr<LNode>> ClusteredHNode::pop_top_k_clusters(int k)
{
  // Build a rank lookup: order[rank] = agent  =>  rank_of[agent] = rank.
  std::vector<uint> rank_of(order.size());
  for (uint rank = 0; rank < order.size(); ++rank)
    rank_of[order[rank]] = rank;

  // Collect (rank, cluster_index) for every non-exhausted cluster tree.
  std::vector<std::pair<uint, size_t>> ranked;
  for (size_t ci = 0; ci < cluster_trees.size(); ++ci) {
    const ClusterTree& ct = cluster_trees[ci];
    if (ct.tree.empty()) continue;
    const uint depth_idx = ct.tree.front()->depth;
    if (depth_idx >= ct.order.size()) continue;
    const uint agent = ct.order[depth_idx];
    const uint rank = (agent < rank_of.size()) ? rank_of[agent] : UINT_MAX;
    ranked.emplace_back(rank, ci);
  }

  if (ranked.empty()) return {};

  // Sort ascending by rank (lower rank = higher priority) and take top k.
  std::sort(ranked.begin(), ranked.end());
  const int take = std::min(k, static_cast<int>(ranked.size()));

  std::vector<std::shared_ptr<LNode>> result;
  result.reserve(take);
  for (int i = 0; i < take; ++i) {
    ClusterTree& ct = cluster_trees[ranked[i].second];
    auto L = ct.tree.front();
    ct.tree.pop();
    expand_cluster_tree(ct, L);
    result.push_back(std::move(L));
  }
  return result;
}

std::shared_ptr<LNode> ClusteredHNode::pop_priority_lnode()
{
  // Build a rank lookup: order[rank] = agent  =>  rank_of[agent] = rank.
  std::vector<uint> rank_of(order.size());
  for (uint rank = 0; rank < order.size(); ++rank)
    rank_of[order[rank]] = rank;

  // Find the non-empty cluster tree whose front LNode will constrain the
  // agent with the lowest rank (= highest priority).
  int best_idx = -1;
  uint best_rank = UINT_MAX;
  for (size_t ci = 0; ci < cluster_trees.size(); ++ci) {
    const ClusterTree& ct = cluster_trees[ci];
    if (ct.tree.empty()) continue;
    const uint depth_idx = ct.tree.front()->depth;
    if (depth_idx >= ct.order.size()) continue;  // tree is fully constrained
    const uint agent = ct.order[depth_idx];
    const uint rank = (agent < rank_of.size()) ? rank_of[agent] : UINT_MAX;
    if (rank < best_rank) { best_rank = rank; best_idx = static_cast<int>(ci); }
  }

  if (best_idx < 0) return nullptr;  // all trees exhausted

  ClusterTree& ct = cluster_trees[best_idx];
  auto L = ct.tree.front();
  ct.tree.pop();
  expand_cluster_tree(ct, L);
  return L;
}

void ClusteredHNode::init_cluster_trees(const std::vector<Cluster>& clusters)
{
  if (trees_initialized) return;
  trees_initialized = true;

  cluster_trees.resize(clusters.size());
  for (size_t ci = 0; ci < clusters.size(); ++ci) {
    ClusterTree& ct = cluster_trees[ci];

    // Build priority order for this cluster: filter HNode::order to members.
    std::unordered_set<uint> agent_set(clusters[ci].agents.begin(),
                                       clusters[ci].agents.end());
    for (uint k : order) {
      if (agent_set.count(k)) ct.order.push_back(k);
    }

    // Seed with an unconstrained root LNode.
    ct.tree.push(std::make_shared<LNode>());
  }
}

Constraints ClusteredHNode::get_constraints(int /*time_window_idx*/,
                                            int cluster_idx)
{
  if (cluster_idx < 0 || static_cast<size_t>(cluster_idx) >= cluster_trees.size())
    return {};

  ClusterTree& ct = cluster_trees[cluster_idx];
  if (ct.tree.empty()) return {};

  // Pop the front LNode.
  auto L = ct.tree.front();
  ct.tree.pop();

  // Expand: add children (constrain the next agent in the cluster's order).
  expand_cluster_tree(ct, L);

  // Extract constraints from the LNode chain.
  Constraints constraints;
  constraints.reserve(L->depth);
  for (const auto* node = L.get();
       node != nullptr && node->parent != nullptr;
       node = node->parent.get()) {
    constraints.emplace_back(node->_who, node->_where);
  }
  std::reverse(constraints.begin(), constraints.end());
  return constraints;
}

bool ClusteredHNode::all_trees_exhausted() const
{
  if (!trees_initialized) return false;
  for (const auto& ct : cluster_trees) {
    if (!ct.tree.empty()) return false;
  }
  return true;
}

void ClusteredHNode::expand_cluster_tree(ClusterTree& ct,
                                         const std::shared_ptr<LNode>& L)
{
  // Each LNode's depth corresponds to the index in the cluster's agent order
  // we should constrain next.
  if (L->depth >= ct.order.size()) return;

  const uint agent_id = ct.order[L->depth];
  auto neighbors = C[agent_id]->neighbor;
  neighbors.push_back(C[agent_id]);  // staying in place is also an option

  for (auto* v : neighbors) {
    ct.tree.push(std::make_shared<LNode>(L, agent_id, v));
  }
}
