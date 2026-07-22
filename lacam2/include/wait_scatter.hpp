#pragma once

#include "scatter.hpp"
#include "collision_table.hpp"
#include "dist_table.hpp"
#include "graph.hpp"
#include "utils.hpp"

struct WaitScatter : IScatter {
  struct Node {
    Vertex *v;
    int g;           // cost-to-come
    int d;           // cost-to-go
    int collisions;
    Node *parent;
    uint32_t tie_breaker;
  };
  const Instance *ins;
  const Deadline *deadline;
  std::mt19937 MT;
  const int verbose;
  const int N;
  const int V_size;
  const int T;  // makespan lower bound
  DistTable *D;
  std::deque<Node> arena = std::deque<Node>();

  const int cost_margin;
  int sum_of_path_length;

  // outcome
  std::vector<Path> paths;
  // agent, vertex-id, next vertex
  std::vector<std::unordered_map<int, Vertex *>> scatter_data;

  const std::unordered_map<int, Vertex *>& operator[](int agent_id) const override
  {
    return scatter_data[agent_id];
  }

  const std::vector<Path>& get_paths() const override { return paths; }

  void seed_from_paths(const std::vector<Path>& new_paths) override
  {
    for (int i = 0; i < N; ++i) CT.clearPath(i, paths[i]);
    paths = new_paths;
    for (int i = 0; i < N; ++i) CT.enrollPath(i, paths[i]);
  }

  // collision data
  CollisionTable CT;

  void construct(int iterations) override;

  template<typename OpenQueue, typename RandFunc>
  void expand(Node* node, int i, Vertex* s_i, int cost_ub,
              std::deque<Node>& arena, OpenQueue& OPEN, RandFunc& fast_rand);

  Path astar(int i, std::vector<int>& CLOSED_cost, std::vector<int>& CLOSED_gen,
             int& current_gen, uint32_t fast_seed);

  WaitScatter(const Instance *_ins, DistTable *_D, const Deadline *_deadline,
              const int seed = 0, int _verbose = 0, int _cost_margin = 2);
};
