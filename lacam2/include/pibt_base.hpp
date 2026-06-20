#pragma once

#include "graph.hpp"
#include "rollout_result.hpp"

// Forward declarations
struct HNode;
struct LNode;

// Abstract interface shared by PIBT and PolicyPIBT.
class PIBTBase {
 public:
  virtual ~PIBTBase() = default;

  virtual bool get_new_config(HNode* H, LNode* L, Config& C_new) = 0;
  virtual uint get_edge_cost(const Config& C1, const Config& C2) const = 0;
  virtual RolloutResult rollout(HNode* H) = 0;
};
