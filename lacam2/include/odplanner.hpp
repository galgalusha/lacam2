/*
 * ODPlanner: Operator Decomposition A* for MAPF
 */
#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "pibt.hpp"
#include "planner.hpp"
#include "utils.hpp"

// LNode subclass that encodes a partial Config as constraints directly,
// without building a linked-list chain. O(1) construction.
struct ConfigLNode : LNode {
  std::vector<uint>   _who_vec;
  Vertices            _where_vec;

  explicit ConfigLNode(const Config& partial_C)
    : LNode(nullptr, 0, nullptr)
  {
    for (uint i = 0; i < partial_C.size(); ++i) {
      if (partial_C[i] != nullptr) {
        _who_vec.push_back(i);
        _where_vec.push_back(partial_C[i]);
      }
    }
    // patch depth (const in base, use placement trick via const_cast)
    const_cast<uint&>(depth) = static_cast<uint>(_who_vec.size());
  }

  std::vector<uint> who()   const override { return _who_vec; }
  Vertices          where() const override { return _where_vec; }
};

#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

// OD A* node.
// A node is "complete" when ordering.size() == N (all agents have been
// assigned a move for this timestep and a fresh ordering is initialised).
// While agents are being assigned one-by-one the node is "temporary"
// and ordering holds only the agents that still need a move.
struct ODNode {
  static uint ODNODE_CNT;

  // Current complete configuration (meaningful for complete nodes; for temp
  // nodes it is the parent complete configuration kept for reference).
  const Config C;

  // The next configuration being built up agent-by-agent.
  Config next_C;

  // For incomplete nodes: the fully-completed config produced by
  // get_new_config using next_C as hard constraints + PIBT for the rest.
  // Empty for complete nodes (use C instead).
  Config completed_C;

  // Ordering of agents still to be assigned in this round.
  // For a complete node this always has size N.
  std::vector<uint> ordering;

  // Cost and heuristic.
  uint g;
  uint h;
  uint f;
  uint depth;

  // Pointer to the nearest ancestor that was a complete node.
  ODNode* parent;

  ODNode(const Config& _C, const Config& _next_C,
         const std::vector<uint>& _ordering, uint _g, uint _h,
         ODNode* _parent);
  ~ODNode() {}

  // Returns true iff this is a complete node (ordering covers all N agents).
  bool isComplete() const;

  // Initialise ordering for a complete node: agents sorted by distance to
  // goal, farthest first.
  void initialize_order(DistTable& D, uint N);
};

// Hash wrapper so complete ODNodes can be stored in a closed list keyed by
// their Config.
struct ODNodeConfigHasher {
  uint operator()(const Config& C) const;
};

class ODPlanner {
public:
  const Instance* ins;
  const Deadline* deadline;
  std::mt19937* MT;
  const int verbose;

  const uint N;      // number of agents
  DistTable D;
  std::unique_ptr<PIBT> pibt;
  uint loop_cnt;
  uint best_f;

  ODPlanner(const Instance* _ins, const Deadline* _deadline,
            std::mt19937* _MT, const int _verbose = 0);
  ~ODPlanner() {}

  Solution solve(std::string& additional_info);

private:
  // Returns PIBT rollout cost as heuristic, or -1 if rollout failed.
  int heuristic(const Config& C);

  // Generate successor ODNodes of the given node and append them to `succs`.
  void expand(ODNode* node, std::vector<ODNode*>& succs);

  // Build the solution path by following parent pointers from a goal node.
  Solution build_solution(ODNode* goal_node) const;
};
