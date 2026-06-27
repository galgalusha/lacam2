#pragma once

#include "cluster_detection_pibt.hpp"
#include "clustered_hnode.hpp"
#include "dist_table.hpp"
#include "instance.hpp"
#include "pibt.hpp"
#include "planner.hpp"
#include "post_processing.hpp"
#include "utils.hpp"

#include <memory>
#include <stack>
#include <unordered_map>

class ClusteredPlanner {
 public:
  const Instance* ins;
  const Deadline* deadline;
  std::mt19937* MT;
  const int verbose;

  const uint N;
  const uint V_size;
  DistTable D;

  // Number of rollouts used to discover temporal clusters.
  const int detection_rollouts;
  // Time window size (in timesteps) for clustering.
  uint time_window;

  ClusteredPlanner(const Instance* _ins, const Deadline* _deadline,
                   std::mt19937* _MT, const int _verbose = 0,
                   int detection_rollouts = 100, uint time_window = 3);

  Solution solve();

 private:
  uint loop_cnt;

  // Return the window index for a given planner depth.
  uint window_of(int depth) const;

  // Get clusters for the window that covers `depth`, formatted as Cluster
  // structs with sequential indices.
  std::vector<Cluster> get_clusters_at(const ClusterDetectionPIBT& cpibt,
                                       int depth) const;

  // Build a solution path from a goal ClusteredHNode.
  Solution build_solution(ClusteredHNode* H_goal) const;

  template <typename... Body>
  void solver_info(int level, Body&&... body)
  {
    if (verbose < level) return;
    std::cout << "elapsed:" << std::setw(6) << elapsed_ms(deadline) << "ms"
              << "  loop_cnt:" << std::setw(8) << loop_cnt
              << "  node_cnt:" << std::setw(8) << HNode::HNODE_CNT << "\t";
    info(level, verbose, std::forward<Body>(body)...);
  }

  uint get_edge_cost(const Config& C1, const Config& C2) const;
  uint get_h_value(const Config& C);
  void rewrite(HNode* H_from, HNode* H_to, HNode* H_goal,
               std::stack<ClusteredHNode*>& OPEN);
};
