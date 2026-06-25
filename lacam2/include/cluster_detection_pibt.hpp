#pragma once

#include "pibt_base.hpp"
#include "planner.hpp"
#include "rollout_result.hpp"

#include <array>
#include <map>
#include <set>
#include <utility>

class ClusterDetectionPIBT : public PIBTBase {
 public:
  ClusterDetectionPIBT(const Instance* _ins, DistTable& _D, std::mt19937* _MT,
               uint _time_window = 1);
  ~ClusterDetectionPIBT() override;

  uint get_edge_cost(const Config& C1, const Config& C2) const override;
  bool get_new_config(HNode* H, LNode* L, Config& C_new) override;
  RolloutResult rollout(HNode* H) override;
  const Agents& agents() const { return A; }

  // time_window: interactions are bucketed by (step / time_window).
  const uint time_window;

  // Per-window accumulated pairwise interactions across all rollouts.
  // Key is window index (step / time_window).
  // Call compute_clusters() after finishing all rollouts.
  std::map<uint, std::set<std::pair<uint, uint>>> interactions;

  // Per-window connected components derived from interactions.
  // Key is window index.
  std::map<uint, std::vector<std::vector<uint>>> clusters;

  void compute_clusters();

  // Returns the cluster partition for the window that contains `time`.
  // Requires compute_clusters() to have been called first.
  const std::vector<std::vector<uint>>& get_clusters(uint time) const;

  void record_interaction(uint a, uint b, uint time);

 private:
  uint current_step;  // set by rollout() before each get_new_config call
  const Instance* ins;
  DistTable& D;
  std::mt19937* MT;
  const uint N;
  const uint V_size;

  std::vector<std::array<Vertex*, 5> > C_next;  // next locations, used in PIBT
  std::vector<float> tie_breakers;              // random values, used in PIBT
  Agents A;
  Agents occupied_now;   // for quick collision checking
  Agents occupied_next;  // for quick collision checking

  bool funcPIBT(Agent* ai);

  // swap operation
  Agent* swap_possible_and_required(Agent* ai);
  bool is_swap_required(const uint pusher, const uint puller,
                        Vertex* v_pusher_origin, Vertex* v_puller_origin);
  bool is_swap_possible(Vertex* v_pusher_origin, Vertex* v_puller_origin);
};
