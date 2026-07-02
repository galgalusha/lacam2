// ClusterDetectionPIBT — copy of PIBT logic for cluster-based variant.
#include "../include/cluster_pibt.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>


ClusterDetectionPIBT::ClusterDetectionPIBT(const Instance* _ins, DistTable& _D, std::mt19937* _MT,
                         uint _time_window)
    : ins(_ins),
      D(_D),
      MT(_MT),
      N(ins->N),
      V_size(ins->G.size()),
      time_window(_time_window),
      current_step(0),
      C_next(N),
      tie_breakers(V_size, 0),
      A(N, nullptr),
      occupied_now(V_size, nullptr),
      occupied_next(V_size, nullptr)
{
  for (auto i = 0; i < N; ++i) A[i] = new Agent(i);
}

ClusterDetectionPIBT::~ClusterDetectionPIBT()
{
  for (auto a : A) delete a;
}

void ClusterDetectionPIBT::record_interaction(uint a, uint b, uint time)
{
  if (a != b) {
    const uint window_idx = time / time_window;
    interactions[window_idx].emplace(std::min(a, b), std::max(a, b));
  }
}

void ClusterDetectionPIBT::compute_clusters()
{
  clusters.clear();
  for (const auto& [window_idx, window_interactions] : interactions) {
    // Union-Find over agents 0..N-1
    std::vector<uint> parent(N);
    std::iota(parent.begin(), parent.end(), 0);

    std::function<uint(uint)> find = [&](uint x) -> uint {
      if (parent[x] != x) parent[x] = find(parent[x]);
      return parent[x];
    };

    for (const auto& [a, b] : window_interactions) {
      uint ra = find(a), rb = find(b);
      if (ra != rb) parent[ra] = rb;
    }

    std::unordered_map<uint, std::vector<uint>> groups;
    for (uint i = 0; i < N; ++i) groups[find(i)].push_back(i);

    std::vector<std::vector<uint>>& window_clusters = clusters[window_idx];
    for (auto& [_, members] : groups) window_clusters.push_back(std::move(members));
  }
}

const std::vector<std::vector<uint>>& ClusterDetectionPIBT::get_clusters(uint time) const
{
  const uint window_idx = time / time_window;
  auto it = clusters.find(window_idx);
  if (it == clusters.end()) {
    static const std::vector<std::vector<uint>> empty;
    return empty;
  }
  return it->second;
}

uint ClusterDetectionPIBT::get_edge_cost(const Config& C1, const Config& C2) const
{
  uint cost = 0;
  for (uint i = 0; i < N; ++i) {
    if (C1[i] != ins->goals[i] || C2[i] != ins->goals[i]) {
      cost += 1;
    }
  }
  return cost;
}

bool ClusterDetectionPIBT::get_new_config(HNode* H, LNode* L, Config& C_new)
{
  if (H == nullptr) return false;

  const auto who = L->who();
  const auto where = L->where();

  // setup cache
  for (auto a : A) {
    // clear previous cache
    if (a->v_now != nullptr && occupied_now[a->v_now->id] == a) {
      occupied_now[a->v_now->id] = nullptr;
    }
    if (a->v_next != nullptr) {
      occupied_next[a->v_next->id] = nullptr;
      a->v_next = nullptr;
    }

    // set occupied now
    a->v_now = H->C[a->id];
    occupied_now[a->v_now->id] = a;
  }

  // add constraints
  for (uint k = 0; k < L->depth; ++k) {
    const auto i = who[k];        // agent
    const auto l = where[k]->id;  // loc

    // check vertex collision
    if (occupied_next[l] != nullptr) return false;
    // check swap collision
    auto l_pre = H->C[i]->id;
    if (occupied_next[l_pre] != nullptr && occupied_now[l] != nullptr &&
        occupied_next[l_pre]->id == occupied_now[l]->id)
      return false;

    // set occupied_next
    A[i]->v_next = where[k];
    occupied_next[l] = A[i];
  }

  // perform PIBT
  for (auto k : H->order) {
    auto a = A[k];
    if (a->v_next == nullptr && !funcPIBT(a)) return false;  // planning failure
  }

  for (auto a : A) C_new[a->id] = a->v_next;

  return true;
}

RolloutResult ClusterDetectionPIBT::rollout(HNode* H)
{
  if (H == nullptr) return {false, 0, 0, {}, {}};
  if (is_same_config(H->C, ins->goals)) return {true, 0, 0, {}};

  LNode unconstrained;
  Config C_new(N, nullptr);
  std::unordered_set<uint> visited;
  ConfigHasher hasher;
  visited.insert(hasher(H->C));

  uint total_cost = 0;
  uint rollout_depth = 0;
  auto current = H;
  std::vector<HNode*> rollout_nodes;
  std::vector<Config> rollout_configs;

  auto cleanup = [&]() {
    for (auto node : rollout_nodes) delete node;
  };

  while (true) {
    current_step = rollout_depth;
    if (!get_new_config(current, &unconstrained, C_new)) {
      cleanup();
      return {false, 0, 0, {}, {}};
    }

    total_cost += get_edge_cost(current->C, C_new);
    rollout_depth += 1;
    rollout_configs.push_back(C_new);

    if (is_same_config(C_new, ins->goals)) {
      cleanup();
      return {true, total_cost, rollout_depth, rollout_configs, {}};
    }

    const auto h = hasher(C_new);
    if (!visited.insert(h).second) {
      cleanup();
      return {false, 0, 0, {}, {}};
    }

    auto next = new HNode(C_new, D, current, 0, 0);
    rollout_nodes.push_back(next);
    current = next;
  }
}

bool ClusterDetectionPIBT::funcPIBT(Agent* ai)
{
  const auto i = ai->id;
  const auto K = ai->v_now->neighbor.size();

  // get candidates for next locations
  for (auto k = 0; k < K; ++k) {
    auto u = ai->v_now->neighbor[k];
    C_next[i][k] = u;
    if (MT != nullptr)
      tie_breakers[u->id] = get_random_float(MT);  // set tie-breaker
  }
  C_next[i][K] = ai->v_now;

  // sort
  std::sort(C_next[i].begin(), C_next[i].begin() + K + 1,
            [&](Vertex* const v, Vertex* const u) {
              return D.get(i, v) + tie_breakers[v->id] <
                     D.get(i, u) + tie_breakers[u->id];
            });

  Agent* swap_agent = swap_possible_and_required(ai);
  if (swap_agent != nullptr) {
    record_interaction(i, swap_agent->id, current_step);  // swap interaction
    std::reverse(C_next[i].begin(), C_next[i].begin() + K + 1);
  }

  // main operation
  for (auto k = 0; k < K + 1; ++k) {
    auto u = C_next[i][k];

    // avoid vertex conflicts
    if (occupied_next[u->id] != nullptr) {
      record_interaction(i, occupied_next[u->id]->id, current_step);
      continue;
    }

    auto& ak = occupied_now[u->id];

    // avoid swap conflicts
    if (ak != nullptr && ak->v_next == ai->v_now) {
      record_interaction(i, ak->id, current_step);
      continue;
    }

    // reserve next location
    occupied_next[u->id] = ai;
    ai->v_next = u;

    // priority inheritance
    if (ak != nullptr && ak != ai && ak->v_next == nullptr) {
      record_interaction(i, ak->id, current_step);
      if (!funcPIBT(ak)) {
        continue;
      }
    }

    // success to plan next one step
    // pull swap_agent when applicable
    if (k == 0 && swap_agent != nullptr && swap_agent->v_next == nullptr &&
        occupied_next[ai->v_now->id] == nullptr) {
      swap_agent->v_next = ai->v_now;
      occupied_next[swap_agent->v_next->id] = swap_agent;
    }
    return true;
  }

  // failed to secure node
  occupied_next[ai->v_now->id] = ai;
  ai->v_next = ai->v_now;
  return false;
}

Agent* ClusterDetectionPIBT::swap_possible_and_required(Agent* ai)
{
  const auto i = ai->id;
  // ai wanna stay at v_now -> no need to swap
  if (C_next[i][0] == ai->v_now) return nullptr;

  // usual swap situation, c.f., case-a, b
  auto aj = occupied_now[C_next[i][0]->id];
  if (aj != nullptr && aj->v_next == nullptr &&
      is_swap_required(ai->id, aj->id, ai->v_now, aj->v_now) &&
      is_swap_possible(aj->v_now, ai->v_now)) {
    return aj;
  }

  // for clear operation, c.f., case-c
  for (auto u : ai->v_now->neighbor) {
    auto ak = occupied_now[u->id];
    if (ak == nullptr || C_next[i][0] == ak->v_now) continue;
    if (is_swap_required(ak->id, ai->id, ai->v_now, C_next[i][0]) &&
        is_swap_possible(C_next[i][0], ai->v_now)) {
      return ak;
    }
  }

  return nullptr;
}

// simulate whether the swap is required
bool ClusterDetectionPIBT::is_swap_required(const uint pusher, const uint puller,
                            Vertex* v_pusher_origin, Vertex* v_puller_origin)
{
  auto v_pusher = v_pusher_origin;
  auto v_puller = v_puller_origin;
  Vertex* tmp = nullptr;
  while (D.get(pusher, v_puller) < D.get(pusher, v_pusher)) {
    auto n = v_puller->neighbor.size();
    // remove agents who need not to move
    for (auto u : v_puller->neighbor) {
      auto a = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && a != nullptr && ins->goals[a->id] == u)) {
        --n;
      } else {
        tmp = u;
      }
    }
    if (n >= 2) return false;  // able to swap
    if (n <= 0) break;
    v_pusher = v_puller;
    v_puller = tmp;
  }

  // judge based on distance
  return (D.get(puller, v_pusher) < D.get(puller, v_puller)) &&
         (D.get(pusher, v_pusher) == 0 ||
          D.get(pusher, v_puller) < D.get(pusher, v_pusher));
}

// simulate whether the swap is possible
bool ClusterDetectionPIBT::is_swap_possible(Vertex* v_pusher_origin, Vertex* v_puller_origin)
{
  auto v_pusher = v_pusher_origin;
  auto v_puller = v_puller_origin;
  Vertex* tmp = nullptr;
  while (v_puller != v_pusher_origin) {  // avoid loop
    auto n = v_puller->neighbor.size();  // count #(possible locations) to pull
    for (auto u : v_puller->neighbor) {
      auto a = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && a != nullptr && ins->goals[a->id] == u)) {
        --n;      // pull-impossible with u
      } else {
        tmp = u;  // pull-possible with u
      }
    }
    if (n >= 2) return true;  // able to swap
    if (n <= 0) return false;
    v_pusher = v_puller;
    v_puller = tmp;
  }
  return false;
}
