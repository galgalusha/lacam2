#include "../include/clustered_planner.hpp"

#include <iostream>

ClusteredPlanner::ClusteredPlanner(const Instance* _ins,
                                   const Deadline* _deadline,
                                   std::mt19937* _MT, const int _verbose,
                                   const Objective _objective,
                                   const float _restart_rate,
                                   uint _time_window)
    : ins(_ins),
      deadline(_deadline),
      MT(_MT),
      verbose(_verbose),
      objective(_objective),
      restart_rate(_restart_rate),
      time_window(_time_window),
      N(_ins->N),
      V_size(_ins->G.size()),
      D(_ins)
{
}

Solution ClusteredPlanner::solve()
{
  ClusterPIBT cpibt(ins, D, MT, time_window);

  // Build initial HNode from starts
  HNode root(ins->starts, D, nullptr, 0, 0);

  for (int i = 0; i < 5000; ++i) {
    cpibt.rollout(&root);
  }

  cpibt.compute_clusters();

  const auto& all_clusters = cpibt.clusters;
  std::cout << "clusters across " << all_clusters.size() << " window(s):\n";
  for (const auto& [window_idx, window_clusters] : all_clusters) {
    const uint window_start = window_idx * time_window;
    const uint window_end = window_start + time_window - 1;
    std::cout << "  window " << window_idx
              << " [t=" << window_start << ".." << window_end << "]"
              << " (" << window_clusters.size() << " cluster(s)):\n";
    for (const auto& cluster : window_clusters) {
      std::cout << "    {";
      for (auto it = cluster.begin(); it != cluster.end(); ++it) {
        if (it != cluster.begin()) std::cout << ", ";
        std::cout << *it;
      }
      std::cout << "}\n";
    }
  }

  return {};
}
