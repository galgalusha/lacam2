/*
 * ODPlanner: Operator Decomposition A* for MAPF
 */
#include "../include/odplanner.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

// ---------------------------------------------------------------------------
// ODNode
// ---------------------------------------------------------------------------

float WEIGHT = 0.7;
uint ODNode::ODNODE_CNT = 0;

ODNode::ODNode(const Config& _C, const Config& _next_C,
               const std::vector<uint>& _ordering, uint _g, uint _h,
               ODNode* _parent)
    : C(_C),
      next_C(_next_C),
      ordering(_ordering),
      g(_g),
      h(_h),
//      f((((int)(_g + _h * WEIGHT) / 10)) * 10),
      f(_g + _h * WEIGHT),
      parent(_parent),
      depth(_parent == nullptr ? 0 : _parent->depth + 1)
{
  ++ODNODE_CNT;
}

bool ODNode::isComplete() const
{
  return ordering.size() == C.size();
}

void ODNode::initialize_order(DistTable& D, uint N)
{
  ordering.resize(N);
  std::iota(ordering.begin(), ordering.end(), 0);
  // Farthest-from-goal agents come first.
  std::sort(ordering.begin(), ordering.end(), [&](uint i, uint j) {
    return D.get(i, C[i]) > D.get(j, C[j]);
  });
}

// ---------------------------------------------------------------------------
// ODNodeConfigHasher
// ---------------------------------------------------------------------------

uint ODNodeConfigHasher::operator()(const Config& C) const
{
  // Same hash strategy used by ConfigHasher in graph.cpp.
  uint seed = 0;
  for (auto* v : C) {
    if (v == nullptr) continue;
    seed ^= v->id + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

// ---------------------------------------------------------------------------
// ODPlanner
// ---------------------------------------------------------------------------

ODPlanner::ODPlanner(const Instance* _ins, const Deadline* _deadline,
                     std::mt19937* _MT, const int _verbose)
    : ins(_ins),
      deadline(_deadline),
      MT(_MT),
      verbose(_verbose),
      N(_ins->N),
      D(_ins),
      pibt(std::make_unique<PIBT>(_ins, D, _MT)),
      loop_cnt(0)
{
}

int ODPlanner::heuristic(const Config& C)
{
  // Use a PIBT rollout from a temporary HNode to estimate cost-to-go.
  // Returns -1 if the rollout failed (successor should be discarded).
  HNode tmp(C, D, nullptr, 0, 0);
  auto result = pibt->rollout(&tmp);
  if (!result.success) return -1;
  return static_cast<int>(result.cost);
}

// ---------------------------------------------------------------------------
// expand
// ---------------------------------------------------------------------------

void ODPlanner::expand(ODNode* node, std::vector<ODNode*>& succs)
{
  // The agent we are assigning a move to in this expansion step.
  const uint agent = node->ordering.front();

  // Build the remaining ordering (all agents after the front).
  std::vector<uint> remaining(node->ordering.begin() + 1,
                              node->ordering.end());

  Vertex* v_now = node->C[agent];

  // Candidate next vertices: wait + up to 4 neighbours.
  std::vector<Vertex*> candidates;
  candidates.push_back(v_now);  // wait
  for (Vertex* nb : v_now->neighbor) candidates.push_back(nb);

  for (Vertex* v_next : candidates) {
    // --- Conflict check: no two agents occupy the same vertex in next_C ---
    bool conflict = false;
    for (uint other = 0; other < N; ++other) {
      if (other == agent) continue;
      // next_C[other] is only set for agents that have already been assigned
      // (i.e. NOT in the current ordering and NOT the current agent).
      // Agents still in `remaining` haven't been assigned yet so we skip them.
      bool other_assigned = true;
      for (uint r : remaining) {
        if (r == other) { other_assigned = false; break; }
      }
      if (!other_assigned) continue;

      if (node->next_C[other] == v_next) { conflict = true; break; }
    }
    if (conflict) continue;

    // Build the updated next_C.
    Config new_next_C = node->next_C;
    new_next_C[agent] = v_next;

    // g increment: +1 unless the agent is waiting on its goal.
    bool waiting_at_goal = (v_next == ins->goals[agent] &&
                            v_next == v_now);
    uint new_g = node->g + (waiting_at_goal ? 0 : 1);

    if (remaining.empty()) {
      // All agents assigned → create a complete node.
      // The complete node's C becomes new_next_C.
      int new_h = heuristic(new_next_C);
      if (new_h < 0) continue;  // rollout failed – discard
    //   if (new_h + new_g >= best_f + 150 - node->depth * 0.25) {
    //     continue;
    //   }      
      auto* succ = new ODNode(new_next_C, Config(N, nullptr),
                              /*ordering=*/{}, new_g, static_cast<uint>(new_h),
                              node);
      // Initialize ordering for the new complete node.      
      succ->initialize_order(D, N);
      succs.push_back(succ);
    } else {
      // Still more agents to assign → create a temp node.
      // Encode already-assigned moves as PIBT constraints via ConfigLNode,
      // then let PIBT complete the config for the rollout.
      ConfigLNode constraints(new_next_C);
      HNode tmp_h(node->C, D, nullptr, 0, 0);
      Config completed(N, nullptr);
      if (!pibt->get_new_config(&tmp_h, &constraints, completed)) continue;

      int new_h = heuristic(completed);
      if (new_h < 0) continue;  // rollout failed – discard

      auto* succ = new ODNode(node->C, new_next_C, remaining,
                              new_g, static_cast<uint>(new_h),
                              node);
      succ->completed_C = completed;
      succs.push_back(succ);
    }
  }
}

// ---------------------------------------------------------------------------
// build_solution
// ---------------------------------------------------------------------------

Solution ODPlanner::build_solution(ODNode* goal_node) const
{
  // Collect complete nodes on the path from root to goal.
  std::vector<ODNode*> path;
  for (ODNode* n = goal_node; n != nullptr; n = n->parent) path.push_back(n);
  std::reverse(path.begin(), path.end());

  Solution sol;
  for (ODNode* n : path) sol.push_back(n->C);
  return sol;
}

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------

Solution ODPlanner::solve(std::string& additional_info)
{
  auto cmp = [](const ODNode* a, const ODNode* b) {
    if (a->f != b->f) return a->f > b->f;
    return a->g < b->g;
  };

  Solution solution;
  uint max_depth = 0;

  // The starting config for the current high-level iteration.
  Config hl_start = ins->starts;

  for (int hl = 0; hl < 100 && !is_expired(deadline); ++hl) {
    // ---- per-iteration data structures ----
    std::priority_queue<ODNode*, std::vector<ODNode*>, decltype(cmp)> OPEN(cmp);
    std::unordered_map<Config, uint, ODNodeConfigHasher> closed;
    std::vector<ODNode*> all_nodes;

    auto make_node = [&](const Config& C, const Config& next_C,
                         const std::vector<uint>& ordering,
                         uint g, uint h, ODNode* parent) -> ODNode* {
      auto* n = new ODNode(C, next_C, ordering, g, h, parent);
      all_nodes.push_back(n);
      return n;
    };

    int init_h_val = heuristic(hl_start);
    if (init_h_val < 0) break;

    auto* root = make_node(hl_start, Config(N, nullptr), {}, 0,
                           static_cast<uint>(init_h_val), nullptr);
    root->initialize_order(D, N);
    OPEN.push(root);

    best_f = root->g + root->h;
    ODNode* best_f_node = root;

    uint inner_cnt = 0;
    while (!OPEN.empty() && !is_expired(deadline) && inner_cnt < 5000) {
      ++loop_cnt;
      ++inner_cnt;
      ODNode* current = OPEN.top();
      OPEN.pop();

      if (current->f < best_f) {
        best_f = current->f;
        best_f_node = current;
        std::cout << "ODPlanner:"
                  << " hl=" << hl
                  << " best_real_f=" << current->h + current->g
                  << " best_f=" << current->f
                  << std::endl;
      }

      if (current->depth > max_depth) max_depth = current->depth;

      if (loop_cnt % 500 == 0) {
        std::cout << "ODPlanner:"
                  << " loop_cnt=" << loop_cnt
                  << " OPEN=" << OPEN.size()
                  << " depth=" << current->depth
                  << " f=" << current->f
                  << " max_depth=" << max_depth << std::endl;
      }

      if (current->isComplete()) {
        auto it = closed.find(current->C);
        if (it != closed.end() && it->second <= current->g) continue;
        closed[current->C] = current->g;

        if (is_same_config(current->C, ins->goals)) {
          solution = build_solution(current);
          for (ODNode* n : all_nodes) delete n;
          goto done;
        }
      }

      std::vector<ODNode*> succs;
      expand(current, succs);
      for (ODNode* s : succs) {
        all_nodes.push_back(s);
        OPEN.push(s);
      }
    }

    {
      // Build the next starting config from the best node found.
      // If the best node is incomplete, use its completed_C (the config
      // produced by get_new_config during expand); otherwise use its C.
      const Config& best_C = (!best_f_node->isComplete() &&
                               !best_f_node->completed_C.empty())
                                 ? best_f_node->completed_C
                                 : best_f_node->C;
      Config next_start = hl_start;
      for (uint i = 0; i < N; ++i) {
        if (best_C[i] != nullptr) next_start[i] = best_C[i];
      }
      std::cout << "ODPlanner: hl=" << hl
                << " advancing to next high-level iteration"
                << " best_f=" << best_f << std::endl;
      hl_start = next_start;
    }

    for (ODNode* n : all_nodes) delete n;
  }

done:
  if (solution.empty() && verbose >= 1) {
    std::cout << "ODPlanner: no solution found within time limit" << std::endl;
  }

  additional_info = "od_loop_cnt=" + std::to_string(loop_cnt) +
                    ",od_node_cnt=" + std::to_string(ODNode::ODNODE_CNT);

  return solution;
}
