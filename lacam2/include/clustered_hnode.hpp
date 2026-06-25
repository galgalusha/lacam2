#pragma once

#include "planner.hpp"

#include <queue>
#include <vector>

// A constraint is an (agent_id, vertex*) pair that the low-level tree
// recommends for a single timestep step.
using Constraints = std::vector<std::pair<uint, Vertex*>>;

// A cluster groups related agents that interact within a time window.
// The index of a Cluster in the per-depth vector serves as its id.
struct Cluster {
  std::vector<uint> agents;
};

// High-level node that maintains a separate low-level search tree per cluster.
// Inherits all fields from HNode (C, depth, priorities, order, g/h/f, etc.)
// The base HNode::search_tree is unused; cluster_trees replaces it.
struct ClusteredHNode : public HNode {
  struct ClusterTree {
    // Agents in priority order for this cluster (subset of HNode::order).
    std::vector<uint> order;
    std::queue<std::shared_ptr<LNode>> tree;
  };

  // One entry per cluster, in the same order as the cluster vector.
  std::vector<ClusterTree> cluster_trees;

  ClusteredHNode(const Config& C, DistTable& D, HNode* parent,
                 uint g, uint h);

  // Initialize cluster_trees from the cluster partition for this depth.
  // Must be called once before get_constraints.  No-op if already called.
  void init_cluster_trees(const std::vector<Cluster>& clusters);

  // Pop one LNode from cluster cluster_idx's tree, expand it, and return its
  // constraints.  Returns an empty Constraints if the tree is exhausted.
  Constraints get_constraints(int time_window_idx, int cluster_idx);

  // True when every cluster tree is exhausted.
  bool all_trees_exhausted() const;

 private:
  bool trees_initialized = false;

  void expand_cluster_tree(ClusterTree& ct,
                           const std::shared_ptr<LNode>& L);
};
