#include "../include/wait_scatter.hpp"
#include "../include/metrics.hpp"

WaitScatter::WaitScatter(const Instance *_ins, DistTable *_D,
                         const Deadline *_deadline, const int seed,
                         int _verbose, int _cost_margin)
    : ins(_ins),
      deadline(_deadline),
      MT(std::mt19937(seed)),
      verbose(_verbose),
      N(ins->N),
      V_size(ins->G.size()),
      T(get_makespan_lower_bound(*ins, *_D) + _cost_margin),
      D(_D),
      cost_margin(_cost_margin),
      sum_of_path_length(0),
      paths(N),
      scatter_data(N),
      CT(ins)
{
}

void WaitScatter::construct(int iterations)
{
  info(1, verbose, deadline, "wait_scatter", "\tinvoked");

  struct Node {
    Vertex *v;
    int g;           // cost-to-come
    int d;           // cost-to-go
    int collisions;
    Node *parent;
    uint32_t tie_breaker;
  };

  auto calc_cost = [](int c, int g, int d) {
    return c * 12 + g + d;
  };

  auto cmp = [&](const Node *a, const Node *b) {
    int cost_a = calc_cost(a->collisions, a->g, a->d);
    int cost_b = calc_cost(b->collisions, b->g, b->d);
    if (cost_a != cost_b) return cost_a > cost_b;
    if (a->collisions != b->collisions) return a->collisions > b->collisions;
    auto f_a = a->g + a->d;
    auto f_b = b->g + b->d;
    if (f_a != f_b) return f_a > f_b;
    return a->tie_breaker < b->tie_breaker;
  };

  // metrics
  auto collision_cnt_last = 0;
  auto paths_prev = std::vector<Path>();

  // main loop
  auto loop = 0;
  while (loop < iterations) {
    ++loop;
    collision_cnt_last = CT.collision_cnt;

    // randomize planning order
    auto order = std::vector<int>(N, 0);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), MT);

    // single-agent path finding for agent-i
    for (int _i = 0; _i < N; ++_i) {
      if (is_expired(deadline)) break;

      const auto i = order[_i];
      const auto cost_ub = D->get(i, ins->starts[i]) + cost_margin;

      if (!paths[i].empty()) sum_of_path_length -= (paths[i].size() - 1);

      // clear cache
      CT.clearPath(i, paths[i]);

      // arena allocator for nodes — pointers remain stable as deque grows
      auto arena = std::deque<Node>();

      // CLOSED maps vertex-id -> best cost found so far
      auto CLOSED = std::vector<int>(V_size, INT_MAX);

      // setup A*
      auto OPEN = std::priority_queue<Node *, std::vector<Node *>,
                                      decltype(cmp)>(cmp);

      const auto s_i = ins->starts[i];
      arena.push_back({s_i, 0, D->get(i, s_i), 0, nullptr, MT()});
      OPEN.push(&arena.back());

      // A*
      auto solved = false;
      Node *goal_node = nullptr;
      while (!OPEN.empty() && !is_expired(deadline)) {
        // pop
        auto node = OPEN.top();
        OPEN.pop();

        const auto v = node->v;
        int current_cost = calc_cost(node->collisions, node->g, node->d);
        bool is_wait = (node->parent != nullptr && node->parent->v == v);

        // Discard if we've found a strictly cheaper way to this vertex
        // and this isn't a wait action (to avoid the early-arrival trap)
        if (current_cost >= CLOSED[v->id] && !is_wait) {
          continue;
        }

        // Update CLOSED only if this spatial arrival is strictly better
        if (current_cost < CLOSED[v->id]) {
          CLOSED[v->id] = current_cost;
        }

        // check goal condition
        if (v == ins->goals[i]) {
          solved = true;
          goal_node = node;
          break;
        }

        const auto g_v = node->g;
        const auto c_v = node->collisions;

        // expand spatial neighbors
        for (auto u : v->neighbor) {
          auto d_u = D->get(i, u);
          if (u != s_i && d_u + g_v + 1 <= cost_ub) {
            arena.push_back({u, g_v + 1, d_u,
                             CT.getCollisionCost(v, u, g_v) + c_v,
                             node, MT()});
            OPEN.push(&arena.back());
          }
        }

        // expand wait action (stay on v for one timestep)
        {
          auto d_v = D->get(i, v);
          if (d_v + g_v + 1 <= cost_ub) {
            arena.push_back({v, g_v + 1, d_v,
                             CT.getCollisionCost(v, v, g_v) + c_v,
                             node, MT()});
            OPEN.push(&arena.back());
          }
        }
      }

      // backtrack via parent pointers
      if (solved && goal_node != nullptr) {
        paths[i].clear();
        auto cur = goal_node;
        while (cur != nullptr) {
          paths[i].push_back(cur->v);
          cur = cur->parent;
        }
        std::reverse(paths[i].begin(), paths[i].end());
      }

      // register to CT & update collision count
      CT.enrollPath(i, paths[i]);
      sum_of_path_length += paths[i].size() - 1;

    } // agent loop

    paths_prev = paths;
    info(4, verbose, deadline, "wait_scatter", "\titer:", loop,
         "\tcollision_cnt:", CT.collision_cnt);

    if (CT.collision_cnt == 0) break;
    if (is_expired(deadline)) break;

  } // epoch loop

  paths = paths_prev;

  // set scatter data
  for (auto i = 0; i < N; ++i) {
    if (paths[i].empty()) continue;
    for (auto t = 0; t < (int)paths[i].size() - 1; ++t) {
      scatter_data[i][paths[i][t]->id] = paths[i][t + 1];
    }
  }

  info(0, verbose, "wait_scatter", "\tcompleted");
}
