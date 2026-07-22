#include "../include/wait_scatter.hpp"
#include "../include/metrics.hpp"

// HYPER PARAMETERS
const int COLLISION_WEIGHT = 12;
const int ASTAR_WEIGHT = 1;

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

  // metrics
  auto collision_cnt_last = 0;
  auto paths_prev = std::vector<Path>();

  auto CLOSED_cost = std::vector<int>(V_size, 0);
  auto CLOSED_gen = std::vector<int>(V_size, 0);
  int current_gen = 0;

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
      current_gen++;

      if (is_expired(deadline)) break;

      const auto i = order[_i];

      if (!paths[i].empty()) sum_of_path_length -= (paths[i].size() - 1);

      // clear cache
      CT.clearPath(i, paths[i]);

      // A*
      auto new_path = astar(i, CLOSED_cost, CLOSED_gen, current_gen, MT());
      if (!new_path.empty()) paths[i] = std::move(new_path);

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

Path WaitScatter::astar(int i,
                        std::vector<int>& CLOSED_cost, std::vector<int>& CLOSED_gen,
                        int& current_gen, uint32_t fast_seed)
{
  const auto cost_ub = D->get(i, ins->starts[i]) + cost_margin;
  const auto s_i = ins->starts[i];

  auto fast_rand = [&fast_seed]() {
    fast_seed ^= fast_seed << 13;
    fast_seed ^= fast_seed >> 17;
    fast_seed ^= fast_seed << 5;
    return fast_seed;
  };

  auto calc_cost = [](int c, int g, int d) {
    return c * COLLISION_WEIGHT + g + d * ASTAR_WEIGHT;
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

  arena.clear();
  auto OPEN = std::priority_queue<Node *, std::vector<Node *>, decltype(cmp)>(cmp);

  arena.push_back({s_i, 0, D->get(i, s_i), 0, nullptr, fast_rand()});
  OPEN.push(&arena.back());

  while (!OPEN.empty() && !is_expired(deadline)) {
    auto node = OPEN.top();
    OPEN.pop();

    const auto v = node->v;
    int current_cost = calc_cost(node->collisions, node->g, node->d);
    bool is_wait = (node->parent != nullptr && node->parent->v == v);

    if (CLOSED_gen[v->id] != current_gen) {
      CLOSED_cost[v->id] = INT_MAX;
      CLOSED_gen[v->id] = current_gen;
    }

    if (current_cost >= CLOSED_cost[v->id] && !is_wait) continue;

    if (current_cost < CLOSED_cost[v->id]) {
      CLOSED_cost[v->id] = current_cost;
    }

    if (v == ins->goals[i]) {
      Path result;
      auto cur = node;
      while (cur != nullptr) {
        result.push_back(cur->v);
        cur = cur->parent;
      }
      std::reverse(result.begin(), result.end());
      return result;
    }

    expand(node, i, s_i, cost_ub, arena, OPEN, fast_rand);
  }

  return {};
}

template<typename OpenQueue, typename RandFunc>
void WaitScatter::expand(Node* node, int i, Vertex* s_i, int cost_ub,
                         std::deque<Node>& arena, OpenQueue& OPEN, RandFunc& fast_rand)
{
  const auto v = node->v;
  const auto g_v = node->g;
  const auto c_v = node->collisions;

  bool collision_detected = false;

  // expand spatial neighbors
  for (auto u : v->neighbor) {
    auto d_u = D->get(i, u);
    if (u != s_i && d_u + g_v + 1 <= cost_ub) {
      int step_collisions = CT.getCollisionCost(v, u, g_v);

      // The Gate: If a spatial move hits traffic, unlock the ability to wait
      if (step_collisions > 0) {
        collision_detected = true;
      }

      arena.push_back({u, g_v + 1, d_u,
                       step_collisions + c_v,
                       node, fast_rand()});
      OPEN.push(&arena.back());
    }
  }

  // expand wait action (stay on v for one timestep) - Semi-Wait-Aware
  if (collision_detected) {
    auto d_v = D->get(i, v);
    if (d_v + g_v + 1 <= cost_ub) {
      arena.push_back({v, g_v + 1, d_v,
                       CT.getCollisionCost(v, v, g_v) + c_v,
                       node, fast_rand()});
      OPEN.push(&arena.back());
    }
  }
}
