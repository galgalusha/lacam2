/*
 * ODPlanner: Operator Decomposition A* for MAPF
 */
#include "../include/odplanner.hpp"
#include "../include/pibt.hpp"
#include "../include/rollout_result.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>


// ---------------------------------------------------------------------------
// ODNode
// ---------------------------------------------------------------------------

float WEIGHT = 1;
uint ODNode::ODNODE_CNT = 0;
int ROLLOUT_NUM = 20;

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
//      pibt(std::make_unique<PIBT>(_ins, D, nullptr)), // no more random
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


RolloutResult get_rollout(Config& C, DistTable& D, PIBT* pibt) {
    HNode tmp(C, D, nullptr, 0, 0);
    std::vector<RolloutResult> rollouts;
    for (int _ = 0; _ < ROLLOUT_NUM; _++) {
        RolloutResult rollout = pibt->rollout(&tmp);
        if (rollout.success) rollouts.push_back(rollout);
    }

    if (rollouts.size() < 5) {
        RolloutResult failed;
        failed.success = false;
        return failed;
    }

    double mean_cost = 0.0;
    for (const auto& r : rollouts) mean_cost += r.cost;
    mean_cost /= rollouts.size();
    double best_diff = std::numeric_limits<double>::max();
    RolloutResult avg_rollout;
    for (const auto& r : rollouts) {
        double diff = std::abs(static_cast<double>(r.cost) - mean_cost);
        if (diff < best_diff) { best_diff = diff; avg_rollout = r; }
    }
    return avg_rollout;
}

// ---------------------------------------------------------------------------
// expand
// ---------------------------------------------------------------------------

void ODPlanner::expand(ODNode* root, ODNode* node, std::vector<ODNode*>& succs)
{
  // The agent we are assigning a move to in this expansion step.
  const uint agent = node->ordering.front();

  // Build the remaining ordering (all agents after the front).
  std::vector<uint> remaining(node->ordering.begin() + 1,
                              node->ordering.end());

  if (remaining.empty()) {
        std::cout << "FOUND COMPLETE" << std::endl;
        return;
  }

  Vertex* v_now = root->C[agent];

  // Candidate next vertices: wait + up to 4 neighbours.
  std::vector<Vertex*> candidates;
  candidates.push_back(v_now);  // wait
  for (Vertex* nb : v_now->neighbor) candidates.push_back(nb);

  for (Vertex* v_next : candidates) {
    bool conflict = is_conflict(agent, remaining, node, v_next);
    if (conflict) continue;

    // Build the updated next_C.
    Config new_next_C = node->next_C;
    new_next_C[agent] = v_next;

    // g increment: +1 unless the agent is waiting on its goal.
    bool waiting_at_goal = (v_next == ins->goals[agent] &&
                            v_next == v_now);
    uint new_g = node->g + (waiting_at_goal ? 0 : 1);

    ConfigLNode constraints(new_next_C);
    HNode tmp_h(root->C, D, nullptr, 0, 0);
    Config completed(N, nullptr);
    if (!pibt->get_new_config(&tmp_h, &constraints, completed)) continue;

    RolloutResult rollout = get_rollout(completed, D, pibt.get());
    if (!rollout.success) continue;

    int cost_to_complete_full_step = pibt->get_edge_cost(root->C, completed);
    int new_h = rollout.cost + cost_to_complete_full_step - new_g;


    // new_g = root->g;
    // new_h = rollout.cost + cost_to_complete_full_step;

    auto* succ = new ODNode(completed, new_next_C, remaining,
                            new_g, static_cast<uint>(new_h),
                            node);

    if (succ->f < best_f) {
        best_f = succ->f;
        succ->rollout = rollout.configs;
    }
                        
    succs.push_back(succ);
  }  
}

bool ODPlanner::is_conflict(const uint agent,
                            std::vector<std::seed_seq::result_type>& remaining,
                            ODNode* node, Vertex* v_next)
{  
  bool conflict = false;
  for (uint other = 0; other < N; ++other) {
    if (other == agent) continue;
    // next_C[other] is only set for agents that have already been assigned
    // (i.e. NOT in the current ordering and NOT the current agent).
    // Agents still in `remaining` haven't been assigned yet so we skip them.
    bool other_assigned = true;
    for (uint r : remaining) {
      if (r == other) {
        other_assigned = false;
        break;
      }
    }
    if (!other_assigned) continue;

    if (node->next_C[other] == v_next) {
      conflict = true;
      break;
    }
    // Swap conflict: other moves to agent's current position while agent
    // moves to other's current position.
    if (node->next_C[other] == node->C[agent] &&
        node->C[other] == v_next) {
      conflict = true;
      break;
    }
  }
  return conflict;
}  

// ---------------------------------------------------------------------------
// build_solution
// ---------------------------------------------------------------------------

Solution ODPlanner::build_solution(ODNode* node)
{
  // Collect prefix: walk back from node to root.
  std::vector<ODNode*> path;
  for (ODNode* n = node; n != nullptr; n = n->parent) path.push_back(n);
  std::reverse(path.begin(), path.end());
  Solution sol;
  for (ODNode* n : path) {
    sol.push_back(n->C);
  }
  std::cout << "Solution prefix size: " << sol.size() << std::endl;
  for (auto& cfg : node->rollout) sol.push_back(cfg);
  return sol;
}

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// get_next_best_c
// ---------------------------------------------------------------------------

ODNode* ODPlanner::get_next_best_c(ODNode* root, uint max_iter)
{
  std::vector<ODNode*> all_nodes;  

  auto cmp = [](const ODNode* a, const ODNode* b) {
    if (a->f != b->f) return a->f > b->f;
    return a->g < b->g;
  };

  int best_f = 999999;
  ODNode* best_f_node = nullptr;

  std::priority_queue<ODNode*, std::vector<ODNode*>, decltype(cmp)> OPEN(cmp);
  OPEN.push(root);
  uint inner_cnt = 0;

  while (!OPEN.empty() && !is_expired(deadline) && inner_cnt < max_iter) {
    ++loop_cnt;
    ++inner_cnt;
    ODNode* current = OPEN.top();
    OPEN.pop();

    if (current->f < best_f && current != root) {
      best_f = current->f;
      best_f_node = current;
      std::cout << "ODPlanner:"
                << " best_f=" << current->f << ", g=" << current->g << ", h=" << current->h  
                << " root g=" << root->g 
                << std::endl;
    }

    if (inner_cnt % 100 == 0) {
      std::cout << "ODPlanner:"
                << " inner_cnt=" << inner_cnt
                << " OPEN=" << OPEN.size()
                << " depth=" << current->depth
                << std::endl;
    }

    std::vector<ODNode*> succs;
    expand(root, current, succs);
    for (ODNode* s : succs) {
      all_nodes.push_back(s);
      OPEN.push(s);
    }
  }

  return best_f_node;
}

// ---------------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------------

Solution ODPlanner::solve(std::string& additional_info)
{
  Solution solution;
  uint max_depth = 0;

  Config hl_start = ins->starts;
  ODNode* root = new ODNode(ins->starts, Config(N, nullptr), {}, 0, 0, nullptr);
  root->initialize_order(D, N);
  root->f = 999999;
  
  ODNode* node = root;
  ODNode* best = root;
  int next_C_idx_to_pick_from_best = 0;
  

  for (int hl = 0; hl < 50 && !is_expired(deadline); ++hl) {

    // auto orig_g = node->g;
    // node->g = 0;
    ODNode* best_next = get_next_best_c(node, 200 + hl * 100);
    // node->g = orig_g;

    if (best_next == nullptr) {
        std::cout << "Cant find next best config" << std::endl;
        break;
    }

//  int new_h = rollout.cost + cost_to_complete_full_step - new_g;

    auto edge_cost = pibt->get_edge_cost(node->C, best_next->C);
    best_next->h += best_next->g - edge_cost; // edge_cost;
    best_next->g = node->g + edge_cost;
    best_next->f = best_next->h + best_next->g;
    best_next->parent = node;
    best_next->next_C = Config(N, nullptr);
    best_next->initialize_order(D, N);
    best_next->depth = 0;

    // if (best_next->f < best->f) {
    //     best = best_next;
    //     next_C_idx_to_pick_from_best = 0;
    // }

    if (best_next->f > node->f && node->rollout.size() > 0) {
            std::cout << "Didn't find better next. Setting next to be next in best rollout after node->C" << std::endl;
            std::cout << "best_f=" << node->f << ", current_f=" << best_next->f << std::endl;
            best_next->C = node->rollout[0];
            auto edge_cost = pibt->get_edge_cost(node->C, best_next->C);
            best_next->h = node->h - edge_cost;
            best_next->g = node->g + edge_cost;
            best_next->f = best_next->h + best_next->g;
            best_next->rollout = std::vector<Config>(node->rollout.begin() + 1, node->rollout.end());
    }

    node = best_next;
    ROLLOUT_NUM = 20 / (hl + 1);

    std::cout << "ODPlanner: hl=" << hl
              << " advancing to next high-level iteration"
              << " f=" << node->f
              << " g=" << node->g
              << " h=" << node->h
              << std::endl;
  }

  solution = build_solution(node);

//   if (solution.empty() && verbose >= 1) {
//     std::cout << "ODPlanner: no solution found within time limit" << std::endl;
//   }

  additional_info = "od_loop_cnt=" + std::to_string(loop_cnt) +
                    ",od_node_cnt=" + std::to_string(ODNode::ODNODE_CNT);

  return solution;
}
