/*
 * lacam-star
 */

#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

#include <chrono>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <unordered_set>


// objective function
enum Objective { OBJ_NONE, OBJ_MAKESPAN, OBJ_SUM_OF_LOSS };
std::ostream& operator<<(std::ostream& os, const Objective objective);

// PIBT agent
struct Agent {
  const uint id;
  Vertex* v_now;   // current location
  Vertex* v_next;  // next location
  Agent(uint _id) : id(_id), v_now(nullptr), v_next(nullptr) {}
};
using Agents = std::vector<Agent*>;

// low-level node
struct LNode {
  std::shared_ptr<LNode> parent;
  uint _who;
  Vertex* _where;
  const uint depth;
  LNode(std::shared_ptr<LNode> parent = nullptr, uint i = 0,
        Vertex* v = nullptr);  // who and where
  std::vector<uint> who() const;
  Vertices where() const;
};

// high-level node
struct HNode {
  static uint HNODE_CNT;  // count #(high-level node)
  const Config C;
  int depth;

  // tree
  HNode* parent;
  std::set<HNode*> neighbor;

  // costs
  uint g;        // g-value (might be updated)
  const uint h;  // h-value
  uint f;        // g + h (might be updated)

  // for low-level search
  std::vector<float> priorities;
  std::vector<uint> order;
  std::queue<std::shared_ptr<LNode>> search_tree;
  uint ll_search;
  bool max_ll_already_decayed;
  float max_ll;

  HNode(const Config& _C, DistTable& D, HNode* _parent, const uint _g,
        const uint _h);
  ~HNode();
  void initialize_order(DistTable& D);
};
using HNodes = std::vector<HNode*>;

class PIBT;

struct Planner {
  static int max_ll;
  static float max_ll_decay;
  const Instance* ins;
  const Deadline* deadline;
  std::mt19937* MT;
  const int verbose;

  // hyper parameters
  const Objective objective;
  const float RESTART_RATE;  // random restart

  // solver utils
  const uint N;       // number of agents
  const uint V_size;  // number o vertices
  DistTable D;
  uint loop_cnt;      // auxiliary
  std::vector<uint64_t> depth_visit_counts;
  std::unique_ptr<PIBT> pibt;
  std::chrono::steady_clock::time_point last_debug_print;

  Planner(const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
          const int _verbose = 0,
          // other parameters
          const Objective _objective = OBJ_NONE,
          const float _restart_rate = 0.001);
  ~Planner();
  Solution solve(std::string& additional_info);
  void expand_lowlevel_tree(HNode* H, const std::shared_ptr<LNode>& L);
  void rewrite(HNode* H_from, HNode* T, HNode* H_goal,
               std::stack<HNode*>& OPEN);
  uint get_h_value(const Config& C);
  void periodic_node_debug(HNode* H, uint loop_count);

  // utilities
  template <typename... Body>
  void solver_info(const int level, Body&&... body)
  {
    if (verbose < level) return;
    std::cout << "elapsed:" << std::setw(6) << elapsed_ms(deadline) << "ms"
              << "  loop_cnt:" << std::setw(8) << loop_cnt
              << "  node_cnt:" << std::setw(8) << HNode::HNODE_CNT << "\t";
    info(level, verbose, (body)...);
  }
};

struct WPlanner : public Planner {
  struct BucketedSuccessor {
    HNode* node;
    uint bucket;
    uint temp_cost;
  };

  WPlanner(const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
           const int _verbose = 0,
           const Objective _objective = OBJ_NONE,
           const float _restart_rate = 0.001)
      : Planner(_ins, _deadline, _MT, _verbose, _objective, _restart_rate)
  {
  }

  std::vector<BucketedSuccessor> get_successors(HNode* H, uint& best_temp_cost,
                                                uint64_t& num_node_gen,
                                                const uint num_expansions);
  Solution solve(std::string& additional_info);
};
