#pragma once

#include "planner.hpp"

#include <array>

class PIBT {
 public:
  struct RolloutResult {
    bool success;
    uint cost;
    uint makespan;
  };

  PIBT(const Instance* _ins, DistTable& _D, std::mt19937* _MT);
  ~PIBT();

  uint get_edge_cost(const Config& C1, const Config& C2) const;
  bool get_new_config(HNode* H, LNode* L, Config& C_new);
  RolloutResult rollout(HNode* H);
  const Agents& agents() const { return A; }

 private:
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
