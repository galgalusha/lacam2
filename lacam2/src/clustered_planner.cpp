#include "../include/clustered_planner.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stack>
#include <unordered_map>

ClusteredPlanner::ClusteredPlanner(const Instance* _ins,
                                   const Deadline* _deadline,
                                   std::mt19937* _MT, const int _verbose,
                                   int _detection_rollouts,
                                   uint _time_window)
    : ins(_ins),
      deadline(_deadline),
      MT(_MT),
      verbose(_verbose),
      N(_ins->N),
      V_size(_ins->G.size()),
      D(_ins),
      detection_rollouts(_detection_rollouts),
      time_window(_time_window),
      loop_cnt(0)
{
}

uint ClusteredPlanner::window_of(int depth) const
{
  return static_cast<uint>(depth) / time_window;
}

std::vector<Cluster> ClusteredPlanner::get_clusters_at(
    const ClusterDetectionPIBT& cpibt, int depth) const
{
  const auto& raw = cpibt.get_clusters(static_cast<uint>(depth));

  // If no interactions were recorded for this window every agent still
  // needs a move, so fall back to one singleton cluster per agent.
  if (raw.empty()) {
    std::vector<Cluster> singletons;
    singletons.reserve(N);
    for (uint i = 0; i < N; ++i) singletons.push_back({{i}});
    return singletons;
  }

  std::vector<Cluster> result;
  result.reserve(raw.size());
  for (const auto& agents : raw) {
    result.push_back({agents});
  }
  return result;
}

Solution ClusteredPlanner::build_solution(ClusteredHNode* H_goal) const
{
  Solution sol;
  for (auto* node = static_cast<HNode*>(H_goal); node != nullptr;
       node = node->parent) {
    sol.push_back(node->C);
  }
  std::reverse(sol.begin(), sol.end());
  return sol;
}

uint ClusteredPlanner::get_edge_cost(const Config& C1, const Config& C2) const
{
  uint cost = 0;
  for (uint i = 0; i < N; ++i) {
    if (C1[i] != ins->goals[i] || C2[i] != ins->goals[i]) {
      cost += 1;
    }
  }
  return cost;
}

uint ClusteredPlanner::get_h_value(const Config& C)
{
  uint cost = 0;
  for (uint i = 0; i < N; ++i) cost += D.get(i, C[i]);
  return cost;
}

void ClusteredPlanner::rewrite(HNode* H_from, HNode* H_to, HNode* H_goal,
                               std::stack<ClusteredHNode*>& OPEN)
{
  // update neighbors
  H_from->neighbor.insert(H_to);

  // Dijkstra update
  std::queue<HNode*> Q({H_from});
  while (!Q.empty()) {
    auto* n_from = Q.front();
    Q.pop();
    for (auto* n_to : n_from->neighbor) {
      auto g_val = n_from->g + get_edge_cost(n_from->C, n_to->C);
      if (g_val < n_to->g) {
        if (n_to == H_goal)
          solver_info(1, "cost update: ", n_to->g, " -> ", g_val);
        n_to->g = g_val;
        n_to->f = n_to->g + n_to->h;
        n_to->parent = n_from;
        n_to->depth = n_from->depth + 1;
        Q.push(n_to);
        if (H_goal != nullptr && n_to->f < H_goal->f)
          OPEN.push(static_cast<ClusteredHNode*>(n_to));
      }
    }
  }
}

Solution ClusteredPlanner::solve()
{
  solver_info(1, "start clustered search");

  // ------------------------------------------------------------------ //
  // Phase 1 – Cluster detection via rollouts.
  // ------------------------------------------------------------------ //
  ClusterDetectionPIBT cpibt(ins, D, MT, time_window);
  {
    HNode root(ins->starts, D, nullptr, 0, 0);
    for (int r = 0; r < detection_rollouts && !is_expired(deadline); ++r) {
      cpibt.rollout(&root);
    }
  }
  cpibt.compute_clusters();

  if (verbose >= 1) {
    std::cout << "clusters across " << cpibt.clusters.size() << " window(s):\n";
    for (const auto& [widx, wcs] : cpibt.clusters) {
      std::cout << "  window " << widx << " (" << wcs.size()
                << " cluster(s))\n";
    }
  }

  // ------------------------------------------------------------------ //
  // Phase 2 – High-level DFS with cluster-aware PIBT.
  // ------------------------------------------------------------------ //
  ClusterAwarePIBT pibt(ins, D, MT);

  using EXPLORED_MAP =
      std::unordered_map<Config, ClusteredHNode*, ConfigHasher>;

  auto OPEN = std::stack<ClusteredHNode*>();
  EXPLORED_MAP EXPLORED;

  auto* H_init = new ClusteredHNode(ins->starts, D, nullptr, 0, 0);
  OPEN.push(H_init);
  EXPLORED[H_init->C] = H_init;

  ClusteredHNode* H_goal = nullptr;

  while (!OPEN.empty() && !is_expired(deadline)) {
    loop_cnt += 1;

    auto* H = OPEN.top();

    // check lower bounds
    if (H_goal != nullptr && H->f >= H_goal->f) {
      OPEN.pop();
      continue;
    }    

    if (H_goal != nullptr && loop_cnt % 5000 == 0) {
      OPEN.push(H_init);
      continue;
    }

    // Initialize cluster trees for this node if not yet done.
    auto clusters = get_clusters_at(cpibt, H->depth);
    // TEST: override with a single cluster containing all agents
    // std::vector<uint> all_agents(N);
    // std::iota(all_agents.begin(), all_agents.end(), 0);
    // clusters = {Cluster{all_agents}};


    H->init_cluster_trees(clusters);

    // If every cluster tree is exhausted, backtrack.
    if (H->all_trees_exhausted()) {
      OPEN.pop();
      continue;
    }

    // ------------------------------------------------------------- //
    // Build C_next by running ClusterAwarePIBT over each cluster.
    // Alternate iteration order per depth to avoid Cluster 0 dominance.
    // ------------------------------------------------------------- //
    Config C_next(N, nullptr);
    pibt.start_new_session();

    // if (H->depth % 2 != 0) {
    //   std::reverse(clusters.begin(), clusters.end());
    // }

    bool pibt_ok = true;
    for (int ci = 0; ci < static_cast<int>(clusters.size()); ++ci) {
      auto constraints =
          H->get_constraints(static_cast<int>(window_of(H->depth)), ci);
      if (!pibt.get_next_config(H, clusters[ci].agents, constraints, &C_next)) {
        pibt_ok = false;
        break;
      }
    }

    if (!pibt_ok) {
//        std::cout << "PIBT failed" << std::endl;
        continue;
    }

    // All agents must have been assigned a next position.
    bool all_set = true;
    for (uint i = 0; i < N; ++i) {
      if (C_next[i] == nullptr) { all_set = false; break; }
    }
    if (!all_set) {
        std::cout << "not all_set" << std::endl;
        continue;
    }

    // ------------------------------------------------------------- //
    // Handle the successor.
    // ------------------------------------------------------------- //
    auto it = EXPLORED.find(C_next);
    if (it != EXPLORED.end()) {
      // Already explored – rewrite costs if a shorter path was found.
      rewrite(H, it->second, H_goal, OPEN);
      auto H_insert = (MT != nullptr && get_random_float(MT) >= 0.001)
                          ? it->second
                          : H_init;
      if (H_goal == nullptr || H_insert->f < H_goal->f) OPEN.push(H_insert);

    } else {
      auto new_g = H->g + get_edge_cost(H->C, C_next);
      auto new_h = get_h_value(C_next);
      auto new_f = new_g + new_h;
      if (H_goal != nullptr && new_f >= H_goal->f)
        continue;
      auto* H2 = new ClusteredHNode(C_next, D, H, new_g, new_h);
      EXPLORED[C_next] = H2;
      H->neighbor.insert(H2);
      OPEN.push(H2);

      // Check goal.
      if (is_same_config(C_next, ins->goals)) {
        H_goal = H2;
        solver_info(1, "found solution. Cost=", H2->g);
      }
    }
  }

  if (H_goal == nullptr) {
    solver_info(1, "no solution found");
    return {};
  }

  return build_solution(H_goal);
}
