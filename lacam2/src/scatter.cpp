#include "../include/scatter.hpp"
#include "../include/metrics.hpp"

SolutionScatter::SolutionScatter(const std::vector<Config>& solution)
{
  if (solution.empty()) return;
  const int N = static_cast<int>(solution[0].size());
  scatter_data.resize(N);
  const int steps = static_cast<int>(solution.size());

  for (int a = 0; a < N; ++a) {
    // 1. Anchor the goal so PIBT knows to stay put when it arrives
    auto final_v = solution.back()[a];
    scatter_data[a][final_v->id] = final_v;

    // 2. Trace BACKWARDS from the end of the path
    for (int t = steps - 2; t >= 0; --t) {
      auto current_v = solution[t][a];
      auto next_v = solution[t + 1][a];

      // 3. Ignore wait actions entirely
      if (current_v != next_v) {
        // 4. Only record the final chronological departure from this vertex.
        // By checking if it's NOT in the map, we lock in the macro-route 
        // and strip out any spatial loops (like Safe Harbors).
        if (scatter_data[a].find(current_v->id) == scatter_data[a].end()) {
          scatter_data[a][current_v->id] = next_v;
        }
      }
    }
  }
}

Scatter::Scatter(const Instance *_ins, DistTable *_D, const Deadline *_deadline,
                 const int seed, int _verbose, int _cost_margin)
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

// vertex, cost-to-come, cost-to-go, collision, parent
using ScatterNode = std::tuple<Vertex *, int, int, int, Vertex *, uint32_t>;

inline Vertex* vertex(ScatterNode &a)       { return std::get<0>(a); }
inline int cost_to_come(ScatterNode &a)     { return std::get<1>(a); }
inline int cost_to_go(ScatterNode &a)       { return std::get<2>(a); }
inline int collision(ScatterNode &a)        { return std::get<3>(a); }
inline Vertex* parent(ScatterNode &a)       { return std::get<4>(a); }
inline uint32_t tie_breaker(ScatterNode &a) { return std::get<5>(a); }

void Scatter::construct(int iterations)
{
  info(1, verbose, deadline, "scatter", "\tinvoked");

  // define path finding utilities
  auto cmp = [&](ScatterNode &a, ScatterNode &b) {
    // collision
//    if (collision(a) != collision(b)) return collision(a) > collision(b);
    auto f_a = cost_to_come(a) + cost_to_go(a);
    auto f_b = cost_to_come(b) + cost_to_go(b);
    int cost_a = collision(a) * 12 + f_a;
    int cost_b = collision(b) * 12 + f_b;
    if (cost_a != cost_b) return cost_a > cost_b;
    if (collision(a) != collision(b)) return collision(a) > collision(b);
    if (f_a != f_b) return f_a > f_b;
    return tie_breaker(a) < tie_breaker(b);
  };
  auto CLOSED = std::vector<Vertex *>(V_size, nullptr);  // parent

  // metrics
  auto collision_cnt_last = 0;
  auto paths_prev = std::vector<Path>();

  // main loop
  auto loop = 0;
//  while (loop < 2 || CT.collision_cnt < collision_cnt_last) {
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

      // setup A*
      auto OPEN = std::priority_queue<ScatterNode, std::vector<ScatterNode>,
                                      decltype(cmp)>(cmp);
      // used with CLOSED, vertex-id list
      const auto s_i = ins->starts[i];
      OPEN.push(std::make_tuple(s_i, 0, D->get(i, s_i), 0, nullptr, MT()));
      auto USED = std::vector<int>();

      // A*
      auto solved = false;
      while (!OPEN.empty() && !is_expired(deadline)) {
        // pop
        auto node = OPEN.top();
        OPEN.pop();

        // check CLOSED list
        const auto v = vertex(node);
        const auto g_v = cost_to_come(node);  // cost-to-come
        const auto c_v = collision(node);  // collision
        if (CLOSED[v->id] != nullptr) continue;
        CLOSED[v->id] = parent(node);  // parent
        USED.push_back(v->id);

        // check goal condition
        if (v == ins->goals[i]) {
          solved = true;
          break;
        }

        // expand
        for (auto u : v->neighbor) {
          auto d_u = D->get(i, u);
          if (u != s_i && CLOSED[u->id] == nullptr &&
              d_u + g_v + 1 <= cost_ub) {
            // insert new node
            OPEN.push(std::make_tuple(
              u, g_v + 1, d_u,
              CT.getCollisionCost(v, u, g_v) + c_v, v, MT()));
          }
        }
      }

      // backtrack
      if (solved) {
        paths[i].clear();
        auto v = ins->goals[i];
        while (v != nullptr) {
          paths[i].push_back(v);
          v = CLOSED[v->id];
        }
        std::reverse(paths[i].begin(), paths[i].end());
      }

      // register to CT & update collision count
      CT.enrollPath(i, paths[i]);
      sum_of_path_length += paths[i].size() - 1;

      // memory management
      for (auto k : USED) CLOSED[k] = nullptr;

    } // agent loop

    paths_prev = paths;
    info(4, verbose, deadline, "scatter", "\titer:", loop,
         "\tcollision_cnt:", CT.collision_cnt);

    if (CT.collision_cnt == 0) break;
    if (is_expired(deadline)) break;

  } // epoch loop

  paths = paths_prev;

  // set scatter data
  for (auto i = 0; i < N; ++i) {
    if (paths[i].empty()) continue;
    for (auto t = 0; t < paths[i].size() - 1; ++t) {
      scatter_data[i][paths[i][t]->id] = paths[i][t + 1];
    }
  }

  info(0, verbose, "scatter", "\tcompleted");
}

