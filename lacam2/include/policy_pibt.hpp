#pragma once

#include "pibt_base.hpp"
#include "planner.hpp"
#include "policy.hpp"
#include "rollout_result.hpp"
#include "scatter.hpp"
#include <array>
#include <memory>

class PolicyPIBT : public PIBTBase {
 public:
  PolicyPIBT(const Instance* _ins, DistTable& _D, std::shared_ptr<Policy> _policy);
  ~PolicyPIBT() override;

  uint get_edge_cost(const Config& C1, const Config& C2) const override;
  bool get_new_config(HNode* H, LNode* L, Config& C_new) override;
  RolloutResult rollout(HNode* H) override;
  RolloutResult rollout(HNode* H, uint max_cost);
  const Agents& agents() const { return A; }
  Scatter *scatter;

 private:
  const Instance* ins;
  DistTable& D;
  std::shared_ptr<Policy> policy;
  const uint N;
  const uint V_size;

  std::vector<std::array<Vertex*, 5>> C_next;
  std::vector<float> tie_breakers;
  Agents A;
  Agents occupied_now;
  Agents occupied_next;

  bool funcPIBT(Agent* ai);

  Agent* swap_possible_and_required(Agent* ai);
  bool is_swap_required(const uint pusher, const uint puller,
                        Vertex* v_pusher_origin, Vertex* v_puller_origin);
  bool is_swap_possible(Vertex* v_pusher_origin, Vertex* v_puller_origin);
};
