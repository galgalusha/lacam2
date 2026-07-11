/*
 * lacam-star
 */

#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "pibt.hpp"
#include "policy.hpp"
#include "scatter.hpp"
#include "utils.hpp"
#include "agent.hpp"
#include "node.hpp"

#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>


// objective function
enum Objective { OBJ_NONE, OBJ_MAKESPAN, OBJ_SUM_OF_LOSS };
std::ostream& operator<<(std::ostream& os, const Objective objective);


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
  Scatter* scatter;
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

// struct WPlanner : public Planner {
//   struct Successor {
//     HNode* node;
//     uint depth;
//     uint temp_cost;
//     std::vector<Config> rollout;
//   };

//   std::unordered_set<uint> seen_states;

//   WPlanner(const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
//            const int _verbose = 0,
//            const Objective _objective = OBJ_NONE,
//            const float _restart_rate = 0.001)
//       : Planner(_ins, _deadline, _MT, _verbose, _objective, _restart_rate)
//   {
//   }

//   using PIBTFactory = std::function<std::unique_ptr<PIBTBase>(std::mt19937*)>;

//   std::vector<Successor> get_successors(HNode* H, uint& best_temp_cost,
//                                         uint64_t& num_node_gen,
//                                         const uint num_expansions,
//                                         const uint how_many,
//                                         const bool save_rollouts = false,
//                                         const PIBTFactory& pibt_factory = nullptr);
//   ScorePolicy create_policy(int num_agents);
//   ScorePolicy create_policy(const Config& start_config, int num_agents);
//   void test_policy(int agent_id);
//   Solution solve(std::string& additional_info);

//   // Returns the HNode at the given depth reachable from this successor.
//   // Walks up parents if depth <= successor.node->depth, or builds HNodes
//   // from the rollout if depth > successor.node->depth.
//   // Any newly created nodes are appended to `created_nodes` (caller owns them).
//   HNode* get_node_at_depth(const Successor& successor, uint depth);
// };
