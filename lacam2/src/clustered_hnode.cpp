#include "../include/clustered_hnode.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

ClusteredHNode::ClusteredHNode(const Config& C, DistTable& D, HNode* parent,
                               uint g, uint h)
    : HNode(C, D, parent, g, h)
{
  // HNode constructor pushed a root LNode into HNode::search_tree.
  // We do not use that tree; cluster_trees is used instead.
  // Drain the base search_tree to keep it empty.
  while (!search_tree.empty()) search_tree.pop();
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
