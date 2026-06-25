#pragma once

#include "clustered_hnode.hpp"
#include "dist_table.hpp"
#include "planner.hpp"

#include <array>
#include <vector>

// A session-aware variant of PIBT designed for clustered planning.
//
// Within a single timestep (session), multiple clusters are processed
// sequentially.  The occupied_next table (and agent v_next assignments) are
// shared across all cluster calls so that agents locked by an earlier cluster
// act as dynamic obstacles for later clusters.
//
// Usage per timestep:
//   pibt->start_new_session();
//   for (each cluster) {
//     auto constraints = H->get_constraints(...);
//     pibt->get_next_config(H, cluster.agents, constraints, &C_next);
//   }
class ClusterAwarePIBT {
 public:
  ClusterAwarePIBT(const Instance* ins, DistTable& D, std::mt19937* MT);
  ~ClusterAwarePIBT();

  // Reset session state (occupied_next, v_next assignments).
  // Call once at the beginning of each timestep before any get_next_config.
  void start_new_session();

  // Advance agents in cluster_agents by one step, respecting:
  //   - positions already committed in *C_next by prior cluster calls (hard),
  //   - constraints from the low-level tree (soft: skipped on conflict).
  // Newly committed positions (including any agents pushed via priority
  // inheritance) are written back into *C_next.
  // Returns false only if PIBT's stay-in-place fallback fails (extremely rare).
  bool get_next_config(HNode* H, const std::vector<uint>& cluster_agents,
                       const Constraints& constraints, Config* C_next);

 private:
  const Instance* ins;
  DistTable& D;
  std::mt19937* MT;
  const uint N;
  const uint V_size;

  // Per-agent candidate buffers (reused across calls, same as PIBT).
  std::vector<std::array<Vertex*, 5>> C_next_buf;
  std::vector<float> tie_breakers;

  // Agent objects (persistent across the planner lifetime).
  Agents A;

  // Collision tables – shared across cluster calls within a session.
  Agents occupied_now;   // indexed by vertex id
  Agents occupied_next;  // indexed by vertex id

  bool session_initialized;  // true after start_new_session(); cleared on next

  bool funcPIBT(Agent* ai);
  Agent* swap_possible_and_required(Agent* ai);
  bool is_swap_required(uint pusher, uint puller,
                        Vertex* v_pusher_origin, Vertex* v_puller_origin);
  bool is_swap_possible(Vertex* v_pusher_origin, Vertex* v_puller_origin);
};
