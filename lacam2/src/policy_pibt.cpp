// PolicyPIBT: PIBT variant with pluggable neighbor selection policy.
#include "../include/policy_pibt.hpp"

#include <algorithm>
#include <unordered_set>


PolicyPIBT::PolicyPIBT(const Instance* _ins, DistTable& _D,
                       std::shared_ptr<Policy> _policy, Scatter* _scatter)
    : ins(_ins),
      D(_D),
      policy(std::move(_policy)),
      N(ins->N),
      V_size(ins->G.size()),
      C_next(N),
      tie_breakers(V_size, 0.0f),
      A(N, nullptr),
      occupied_now(V_size, nullptr),
      occupied_next(V_size, nullptr),
      scatter(_scatter)
{
  for (auto i = 0; i < N; ++i) A[i] = new Agent(i);
}

PolicyPIBT::~PolicyPIBT()
{
  for (auto a : A) delete a;
}

uint PolicyPIBT::get_edge_cost(const Config& C1, const Config& C2) const
{
  uint cost = 0;
  for (uint i = 0; i < N; ++i) {
    if (C1[i] != ins->goals[i] || C2[i] != ins->goals[i]) {
      cost += 1;
    }
  }
  return cost;
}

bool PolicyPIBT::get_new_config(HNode* H, LNode* L, Config& C_new)
{
  if (H == nullptr) return false;

  const auto who = L->who();
  const auto where = L->where();

  // setup cache
  for (auto a : A) {
    if (a->v_now != nullptr && occupied_now[a->v_now->id] == a) {
      occupied_now[a->v_now->id] = nullptr;
    }
    if (a->v_next != nullptr) {
      occupied_next[a->v_next->id] = nullptr;
      a->v_next = nullptr;
    }

    a->v_now = H->C[a->id];
    occupied_now[a->v_now->id] = a;
  }

  // add constraints
  for (uint k = 0; k < L->depth; ++k) {
    const auto i = who[k];
    const auto l = where[k]->id;

    if (occupied_next[l] != nullptr) return false;
    auto l_pre = H->C[i]->id;
    if (occupied_next[l_pre] != nullptr && occupied_now[l] != nullptr &&
        occupied_next[l_pre]->id == occupied_now[l]->id)
      return false;

    A[i]->v_next = where[k];
    occupied_next[l] = A[i];
  }

  // perform PIBT
  for (auto k : H->order) {
    auto a = A[k];
    if (a->v_next == nullptr && !funcPIBT(a)) return false;
  }

  for (auto a : A) C_new[a->id] = a->v_next;

  return true;
}

RolloutResult PolicyPIBT::rollout(HNode* H)
{
  return rollout(H, UINT_MAX);
}

RolloutResult PolicyPIBT::rollout(HNode* H, uint max_cost)
{
  if (H == nullptr) return {false, 0, 0, {}};
  if (is_same_config(H->C, ins->goals)) return {true, 0, 0, {}};

  // heuristic: sum of distances to goals
  auto heuristic = [&](const Config& C) -> uint {
    uint h = 0;
    for (uint agent = 0; agent < N; ++agent) h += D.get(agent, C[agent]);
    return h;
  };

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

  rollout_configs.push_back(H->C);

  auto cleanup = [&]() {
    for (auto node : rollout_nodes) delete node;
  };

  while (true) {
    if (!get_new_config(current, &unconstrained, C_new)) {
      cleanup();
      return {false, 0, 0, {}};
    }

    total_cost += get_edge_cost(current->C, C_new);
    rollout_depth += 1;
    rollout_configs.push_back(C_new);

    // Prune if cost already exceeds bound or can't possibly improve.
    if (total_cost >= max_cost || total_cost + heuristic(C_new) >= max_cost) {
      cleanup();
      return {false, 0, 0, {}};
    }

    if (is_same_config(C_new, ins->goals)) {
      cleanup();
      return {true, total_cost, rollout_depth, rollout_configs};
    }

    const auto h = hasher(C_new);
    if (!visited.insert(h).second) {
      cleanup();
      return {false, 0, 0, {}};
    }

    auto next = new HNode(C_new, D, current, 0, 0);
    rollout_nodes.push_back(next);
    current = next;
  }
}

bool PolicyPIBT::funcPIBT(Agent* ai)
{
  const auto i = ai->id;
  const auto K = ai->v_now->neighbor.size();

  // collect neighbors (including self-stay)
  for (auto k = 0; k < K; ++k) C_next[i][k] = ai->v_now->neighbor[k];
  C_next[i][K] = ai->v_now;

  // set tie-breakers in [-0.9, 0] for each neighbor, written into tie_breakers[v->id]
  policy->set_tie_breakers(i, ai->v_now, C_next[i], K + 1, tie_breakers);

  // sort by D + tie_breaker (lower = preferred)
  std::sort(C_next[i].begin(), C_next[i].begin() + K + 1,
            [&](Vertex* const v, Vertex* const u) {
              return D.get(i, v) + tie_breakers[v->id] < D.get(i, u) + tie_breakers[u->id];
            });

  Agent* swap_agent = swap_possible_and_required(ai);
  if (swap_agent != nullptr)
    std::reverse(C_next[i].begin(), C_next[i].begin() + K + 1);

  // main operation
  for (auto k = 0; k < K + 1; ++k) {
    auto u = C_next[i][k];

    if (occupied_next[u->id] != nullptr) continue;

    auto& ak = occupied_now[u->id];

    if (ak != nullptr && ak->v_next == ai->v_now) continue;

    occupied_next[u->id] = ai;
    ai->v_next = u;

    if (ak != nullptr && ak != ai && ak->v_next == nullptr) {
      if (!funcPIBT(ak)) {
        continue;
      }
    }

    if (k == 0 && swap_agent != nullptr && swap_agent->v_next == nullptr &&
        occupied_next[ai->v_now->id] == nullptr) {
      swap_agent->v_next = ai->v_now;
      occupied_next[swap_agent->v_next->id] = swap_agent;
    }
    return true;
  }

  occupied_next[ai->v_now->id] = ai;
  ai->v_next = ai->v_now;
  return false;
}

Agent* PolicyPIBT::swap_possible_and_required(Agent* ai)
{
  const auto i = ai->id;
  if (C_next[i][0] == ai->v_now) return nullptr;

  auto aj = occupied_now[C_next[i][0]->id];
  if (aj != nullptr && aj->v_next == nullptr &&
      is_swap_required(ai->id, aj->id, ai->v_now, aj->v_now) &&
      is_swap_possible(aj->v_now, ai->v_now)) {
    return aj;
  }

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

bool PolicyPIBT::is_swap_required(const uint pusher, const uint puller,
                                   Vertex* v_pusher_origin,
                                   Vertex* v_puller_origin)
{
  auto v_pusher = v_pusher_origin;
  auto v_puller = v_puller_origin;
  Vertex* tmp = nullptr;
  while (D.get(pusher, v_puller) < D.get(pusher, v_pusher)) {
    auto n = v_puller->neighbor.size();
    for (auto u : v_puller->neighbor) {
      auto a = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && a != nullptr && ins->goals[a->id] == u)) {
        --n;
      } else {
        tmp = u;
      }
    }
    if (n >= 2) return false;
    if (n <= 0) break;
    v_pusher = v_puller;
    v_puller = tmp;
  }

  return (D.get(puller, v_pusher) < D.get(puller, v_puller)) &&
         (D.get(pusher, v_pusher) == 0 ||
          D.get(pusher, v_puller) < D.get(pusher, v_pusher));
}

bool PolicyPIBT::is_swap_possible(Vertex* v_pusher_origin,
                                   Vertex* v_puller_origin)
{
  auto v_pusher = v_pusher_origin;
  auto v_puller = v_puller_origin;
  Vertex* tmp = nullptr;
  while (v_puller != v_pusher_origin) {
    auto n = v_puller->neighbor.size();
    for (auto u : v_puller->neighbor) {
      auto a = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && a != nullptr && ins->goals[a->id] == u)) {
        --n;
      } else {
        tmp = u;
      }
    }
    if (n >= 2) return true;
    if (n <= 0) return false;
    v_pusher = v_puller;
    v_puller = tmp;
  }
  return false;
}
