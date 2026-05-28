#include "../include/planner.hpp"

#include <algorithm>
#include <numeric>

uint HNode::HNODE_CNT = 0;

// for high-level
HNode::HNode(const Config& _C, DistTable& D, HNode* _parent, const uint _g,
             const uint _h)
    : C(_C),
      parent(_parent),
      neighbor(),
      g(_g),
      h(_h),
      h_cbs(0),
      f(g + h),
      priorities(C.size()),
      order(C.size(), 0),
      constraint_order(C.size(), 0),
      search_tree(std::queue<std::shared_ptr<LNode>>()),
      ll_search(0)
{
  ++HNODE_CNT;

  auto root = std::make_shared<LNode>();
  search_tree.push(root);
  ll_search += 1;

  // update neighbor
  if (parent != nullptr) parent->neighbor.insert(this);

  initialize_order(D);
  initialize_constraint_order();
}

HNode::~HNode()
{
}

void HNode::initialize_order(DistTable& D)
{
  const auto N = C.size();

  // set priorities (original logic)
  if (parent == nullptr) {
    // initialize
    for (uint i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
  } else {
    // dynamic priorities, akin to PIBT
    for (size_t i = 0; i < N; ++i) {
      if (D.get(i, C[i]) != 0) {
        priorities[i] = parent->priorities[i] + 1;
      } else {
        priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
      }
    }
  }

  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](uint i, uint j) {
    return priorities[i] > priorities[j];
  });
}

void HNode::initialize_constraint_order()
{
  constraint_order = order;
  return;
  const auto N = C.size();

  // Cluster order is primary key; original priority is secondary key.
  auto cluster_rank = std::vector<uint>(N, N);
  if (parent != nullptr && !parent->pibt_clusters.empty()) {
    for (const auto& cluster : parent->pibt_clusters) {
      for (size_t idx = 0; idx < cluster.size(); ++idx) {
        const auto agent_id = cluster[idx];
        if (agent_id >= N) continue;
        const auto rank_in_cluster = static_cast<uint>(idx);
        if (cluster_rank[agent_id] > rank_in_cluster) {
          cluster_rank[agent_id] = rank_in_cluster;
        }
      }
    }
  }

  // Primary key: cluster rank. Tie-breaker: existing PIBT launch order.
  constraint_order = order;
  std::stable_sort(constraint_order.begin(), constraint_order.end(),
                   [&](uint i, uint j) {
                     return cluster_rank[i] < cluster_rank[j];
                   });
}
