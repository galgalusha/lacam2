// Session-aware PIBT for clustered planning.
#include "../include/cluster_aware_pibt.hpp"

#include <algorithm>
#include <unordered_set>

ClusterAwarePIBT::ClusterAwarePIBT(const Instance* _ins, DistTable& _D,
                                   std::mt19937* _MT)
    : ins(_ins),
      D(_D),
      MT(_MT),
      N(ins->N),
      V_size(ins->G.size()),
      C_next_buf(N),
      tie_breakers(V_size, 0.0f),
      A(N, nullptr),
      occupied_now(V_size, nullptr),
      occupied_next(V_size, nullptr),
      session_initialized(false)
{
  for (uint i = 0; i < N; ++i) A[i] = new Agent(i);
}

ClusterAwarePIBT::~ClusterAwarePIBT()
{
  for (auto* a : A) delete a;
}

void ClusterAwarePIBT::start_new_session()
{
  // Clear committed-next positions.
  for (uint vid = 0; vid < V_size; ++vid) occupied_next[vid] = nullptr;
  // Reset per-agent state.
  for (uint i = 0; i < N; ++i) A[i]->v_next = nullptr;
  // occupied_now will be (re)populated on the first get_next_config call.
  session_initialized = false;
}

bool ClusterAwarePIBT::get_next_config(HNode* H,
                                       const std::vector<uint>& cluster_agents,
                                       const Constraints& constraints,
                                       Config* C_next)
{
  if (H == nullptr) return false;

  // ------------------------------------------------------------------ //
  // 1. Setup occupied_now from H->C.  This is idempotent within a
  //    session (H->C does not change), so we redo it every cluster call
  //    to keep the table consistent.
  // ------------------------------------------------------------------ //
  if (!session_initialized) {
    for (uint vid = 0; vid < V_size; ++vid) occupied_now[vid] = nullptr;
    for (auto* a : A) {
      a->v_now = H->C[a->id];
      occupied_now[a->v_now->id] = a;
    }
    session_initialized = true;
  }

  // ------------------------------------------------------------------ //
  // 2. Re-sync occupied_next from *C_next (positions committed by prior
  //    cluster calls in this session).
  // ------------------------------------------------------------------ //
  for (uint i = 0; i < N; ++i) {
    Vertex* committed = (*C_next)[i];
    if (committed != nullptr) {
      A[i]->v_next = committed;
      occupied_next[committed->id] = A[i];
    }
  }

  // ------------------------------------------------------------------ //
  // 3. Apply soft constraints from the low-level tree.
  //    A constraint is skipped (soft) when it conflicts with an already
  //    committed position in occupied_next.
  // ------------------------------------------------------------------ //
  for (const auto& [agent_id, vertex] : constraints) {
    // Skip if this agent is already committed.
    if (A[agent_id]->v_next != nullptr) continue;

    const uint l = vertex->id;

    // Vertex already taken by another committed agent -> skip.
    if (occupied_next[l] != nullptr) continue;

    // Swap conflict with a committed agent -> skip.
    const uint l_pre = H->C[agent_id]->id;
    if (occupied_next[l_pre] != nullptr && occupied_now[l] != nullptr &&
        occupied_next[l_pre]->id == occupied_now[l]->id)
      continue;

    // Apply the constraint.
    A[agent_id]->v_next = vertex;
    occupied_next[l] = A[agent_id];
  }

  // ------------------------------------------------------------------ //
  // 4. Run PIBT for each agent in cluster_agents that is not yet locked,
  //    iterating in H->order (priority-sorted) to match PIBT behaviour.
  // ------------------------------------------------------------------ //
  const std::unordered_set<uint> cluster_set(cluster_agents.begin(),
                                             cluster_agents.end());
  for (uint k : H->order) {
    if (!cluster_set.count(k)) continue;
    auto* a = A[k];
    if (a->v_next == nullptr && !funcPIBT(a)) return false;
  }

  // ------------------------------------------------------------------ //
  // 5. Write all newly committed positions back to *C_next (including any
  //    agents pushed via priority inheritance from outside the cluster).
  // ------------------------------------------------------------------ //
  for (uint i = 0; i < N; ++i) {
    if (A[i]->v_next != nullptr) {
      (*C_next)[i] = A[i]->v_next;
    }
  }

  return true;
}

// ======================================================================
// funcPIBT and helpers – identical to PIBT except they use the shared
// occupied_now / occupied_next tables and C_next_buf member.
// ======================================================================

bool ClusterAwarePIBT::funcPIBT(Agent* ai)
{
  const uint i = ai->id;
  const uint K = static_cast<uint>(ai->v_now->neighbor.size());

  for (uint k = 0; k < K; ++k) {
    auto* u = ai->v_now->neighbor[k];
    C_next_buf[i][k] = u;
    if (MT != nullptr) tie_breakers[u->id] = get_random_float(MT);
  }
  C_next_buf[i][K] = ai->v_now;

  std::sort(C_next_buf[i].begin(), C_next_buf[i].begin() + K + 1,
            [&](Vertex* v, Vertex* u) {
              return D.get(i, v) + tie_breakers[v->id] <
                     D.get(i, u) + tie_breakers[u->id];
            });

  Agent* swap_agent = swap_possible_and_required(ai);
  if (swap_agent != nullptr)
    std::reverse(C_next_buf[i].begin(), C_next_buf[i].begin() + K + 1);

  for (uint k = 0; k < K + 1; ++k) {
    auto* u = C_next_buf[i][k];

    if (occupied_next[u->id] != nullptr) continue;

    auto* ak = occupied_now[u->id];
    if (ak != nullptr && ak->v_next == ai->v_now) continue;

    occupied_next[u->id] = ai;
    ai->v_next = u;

    if (ak != nullptr && ak != ai && ak->v_next == nullptr) {
      if (!funcPIBT(ak)) continue;
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

Agent* ClusterAwarePIBT::swap_possible_and_required(Agent* ai)
{
  const uint i = ai->id;
  if (C_next_buf[i][0] == ai->v_now) return nullptr;

  auto* aj = occupied_now[C_next_buf[i][0]->id];
  if (aj != nullptr && aj->v_next == nullptr &&
      is_swap_required(ai->id, aj->id, ai->v_now, aj->v_now) &&
      is_swap_possible(aj->v_now, ai->v_now)) {
    return aj;
  }

  for (auto* u : ai->v_now->neighbor) {
    auto* ak = occupied_now[u->id];
    if (ak == nullptr || C_next_buf[i][0] == ak->v_now) continue;
    if (is_swap_required(ak->id, ai->id, ai->v_now, C_next_buf[i][0]) &&
        is_swap_possible(C_next_buf[i][0], ai->v_now)) {
      return ak;
    }
  }

  return nullptr;
}

bool ClusterAwarePIBT::is_swap_required(uint pusher, uint puller,
                                        Vertex* v_pusher_origin,
                                        Vertex* v_puller_origin)
{
  auto* v_pusher = v_pusher_origin;
  auto* v_puller = v_puller_origin;
  Vertex* tmp = nullptr;
  while (D.get(pusher, v_puller) < D.get(pusher, v_pusher)) {
    uint n = v_puller->neighbor.size();
    for (auto* u : v_puller->neighbor) {
      auto* a = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && a != nullptr &&
           ins->goals[a->id] == u)) {
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

bool ClusterAwarePIBT::is_swap_possible(Vertex* v_pusher_origin,
                                        Vertex* v_puller_origin)
{
  auto* v_pusher = v_pusher_origin;
  auto* v_puller = v_puller_origin;
  Vertex* tmp = nullptr;
  while (v_puller != v_pusher_origin) {
    uint n = v_puller->neighbor.size();
    for (auto* u : v_puller->neighbor) {
      auto* a = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && a != nullptr &&
           ins->goals[a->id] == u)) {
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
