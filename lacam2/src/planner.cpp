#include "../include/planner.hpp"
#include "../include/pibt.hpp"
#include "../include/pibt_base.hpp"
#include "../include/policy_pibt.hpp"
#include <chrono>
#include <algorithm>

LNode::LNode(std::shared_ptr<LNode> parent, uint i, Vertex* v)
  : parent(parent), _who(i), _where(v), depth(parent == nullptr ? 0 : parent->depth + 1)
{
}

std::vector<uint> LNode::who() const
{
  std::vector<uint> result;
  result.reserve(depth);

  for (const auto* node = this; node != nullptr && node->parent != nullptr;
     node = node->parent.get()) {
    result.push_back(node->_who);
  }

  std::reverse(result.begin(), result.end());
  return result;
}

Vertices LNode::where() const
{
  Vertices result;
  result.reserve(depth);

  for (const auto* node = this; node != nullptr && node->parent != nullptr;
     node = node->parent.get()) {
    result.push_back(node->_where);
  }

  std::reverse(result.begin(), result.end());
  return result;
}

int Planner::max_ll = -1;
float Planner::max_ll_decay = 1.0f;

Planner::Planner(const Instance* _ins, const Deadline* _deadline,
                 std::mt19937* _MT, const int _verbose,
                 const Objective _objective, const float _restart_rate)
    : ins(_ins),
      deadline(_deadline),
      MT(_MT),
      verbose(_verbose),
      objective(_objective),
      RESTART_RATE(_restart_rate),
      N(ins->N),
      V_size(ins->G.size()),
      D(DistTable(ins)),
      loop_cnt(0),
      depth_visit_counts(1, 0),
      pibt(std::unique_ptr<PIBT>(std::make_unique<PIBT>(ins, D, MT))),
      last_debug_print(std::chrono::steady_clock::now())
{
}

Planner::~Planner() {}


Solution Planner::solve(std::string& additional_info)
{
  info(1, verbose, deadline, "start computing SUO");
  auto scatter_deadline =
      Deadline(deadline == nullptr
                   ? INT_MAX
                   : (deadline->time_limit_ms - elapsed_ms(deadline)) / 2);
  int SCATTER_MARGIN = 10;                 
  scatter = new Scatter(ins, &D, &scatter_deadline, 3, verbose - 4, SCATTER_MARGIN);

  scatter->construct();
  info(1, verbose, deadline, "finish computing SUO",
       ", collision count: ", scatter->CT.collision_cnt,
       ", scatter margin: ", scatter->cost_margin,
       ", sum_of_path_length: ", scatter->sum_of_path_length);
  this->pibt->scatter = scatter;

  solver_info(1, "start search");

  // setup search
  auto OPEN = std::stack<HNode*>();
  auto EXPLORED = std::unordered_map<Config, HNode*, ConfigHasher>();
  // insert initial node, 'H': high-level node
  auto H_init = new HNode(ins->starts, D, nullptr, 0, get_h_value(ins->starts));
  OPEN.push(H_init);
  EXPLORED[H_init->C] = H_init;

  std::vector<Config> solution;
  auto C_new = Config(N, nullptr);  // for new configuration
  HNode* H_goal = nullptr;          // to store goal node


  // DFS
  while (!OPEN.empty() && !is_expired(deadline)) {
    loop_cnt += 1;

    // do not pop here!
    auto H = OPEN.top();  // high-level node
    if (static_cast<size_t>(H->depth) >= depth_visit_counts.size()) {
      depth_visit_counts.resize(H->depth + 1, 0);
    }
    depth_visit_counts[H->depth] += 1;
    // periodic_node_debug(H, loop_cnt);

    // low-level search end
    if (H->search_tree.empty()) {
      OPEN.pop();
      continue;
    }

    // check lower bounds
    if (H_goal != nullptr && H->f >= H_goal->f) {
      OPEN.pop();
      continue;
    }

    // check goal condition
    if (H_goal == nullptr && is_same_config(H->C, ins->goals)) {
      H_goal = H;
      solver_info(1, "found solution, cost: ", H->g);
      if (objective == OBJ_NONE) break;
      continue;
    }

    if (H->max_ll > -1 && static_cast<float>(H->ll_search) > H->max_ll) {
      if (H->parent != nullptr && !H->parent->max_ll_already_decayed) {
        H->parent->max_ll_already_decayed = true;
        H->parent->max_ll = std::max(H->max_ll * Planner::max_ll_decay, 2.0f);
      }
      OPEN.pop();
      continue;
    }

    if (H_goal != nullptr && loop_cnt % 1000 == 0) {
      // std::cout << "restart depth_visit_counts:";
      // for (size_t start = 0; start < depth_visit_counts.size();) {
      //   const auto value = depth_visit_counts[start];
      //   size_t end = start + 1;
      //   while (end < depth_visit_counts.size() && depth_visit_counts[end] == value) {
      //     ++end;
      //   }

      //   std::cout << " d" << start;
      //   if (end - start > 1) {
      //     std::cout << "_" << (end - 1);
      //   }
      //   std::cout << "=" << value;

      //   start = end;
      // }
      // std::cout << std::endl;
      // std::fill(depth_visit_counts.begin(), depth_visit_counts.end(), 0);
      OPEN.push(H_init);
      continue;
    }

    // create successors at the low-level search
    auto L = H->search_tree.front();
    H->search_tree.pop();    
    H->ll_search += 1;

    expand_lowlevel_tree(H, L);

    // create successors at the high-level search
    const auto res = pibt->get_new_config(H, L.get(), C_new);
    if (!res) continue;

    // check explored list
    const auto iter = EXPLORED.find(C_new);
    if (iter != EXPLORED.end()) {
      // case found
      rewrite(H, iter->second, H_goal, OPEN);

      // re-insert or random-restart
      auto H_insert = (MT != nullptr && get_random_float(MT) >= RESTART_RATE)
                          ? iter->second
                          : H_init;
      if (H_goal == nullptr || H_insert->f < H_goal->f) OPEN.push(H_insert);
    } else {
      // insert new search node
      auto new_g = H->g + pibt->get_edge_cost(H->C, C_new);
      auto new_h = get_h_value(C_new);
      auto new_f = new_g + new_h;
      if (H_goal != nullptr && new_f >= H_goal->f)
        continue;
      const auto H_new = new HNode(C_new, D, H, new_g, new_h);
      H->neighbor.insert(H_new);
      EXPLORED[H_new->C] = H_new;
      OPEN.push(H_new);
    }
  }

  // backtrack
  if (H_goal != nullptr) {
    auto H = H_goal;
    while (H != nullptr) {
      solution.push_back(H->C);
      H = H->parent;
    }
    std::reverse(solution.begin(), solution.end());
  }

  // print result
  if (H_goal != nullptr && OPEN.empty()) {
    solver_info(1, "solved optimally, objective: ", objective);
  } else if (H_goal != nullptr) {
    solver_info(1, "solved sub-optimally, objective: ", objective);
  } else if (OPEN.empty()) {
    solver_info(1, "no solution");
  } else {
    solver_info(1, "timeout");
  }

  // logging
  additional_info +=
      "optimal=" + std::to_string(H_goal != nullptr && OPEN.empty()) + "\n";
  additional_info += "objective=" + std::to_string(objective) + "\n";
  additional_info += "loop_cnt=" + std::to_string(loop_cnt) + "\n";
  additional_info += "num_node_gen=" + std::to_string(EXPLORED.size()) + "\n";

  // memory management
  for (auto itr : EXPLORED) delete itr.second;

  return solution;
}

void Planner::rewrite(HNode* H_from, HNode* H_to, HNode* H_goal,
                      std::stack<HNode*>& OPEN)
{
  // update neighbors
  H_from->neighbor.insert(H_to);

  // Dijkstra update
  std::queue<HNode*> Q({H_from});  // queue is sufficient
  while (!Q.empty()) {
    auto n_from = Q.front();
    Q.pop();
    for (auto n_to : n_from->neighbor) {
      auto g_val = n_from->g + pibt->get_edge_cost(n_from->C, n_to->C);
      if (g_val < n_to->g) {
        if (n_to == H_goal)
          solver_info(1, "depth(H): ", H_from->depth, ", depth(G): ",H_goal->depth, ", cost update: ", n_to->g, " -> ", g_val, " from depth ", H_from->depth);
        n_to->g = g_val;
        n_to->f = n_to->g + n_to->h;
        n_to->parent = n_from;
        n_to->depth = n_from->depth + 1;
        // n_to->incoming_pibt_clusters = n_from->pibt_clusters;
        Q.push(n_to);
        if (H_goal != nullptr && n_to->f < H_goal->f) OPEN.push(n_to);
      }
    }
  }
}

uint Planner::get_h_value(const Config& C)
{
  uint cost = 0;
  if (objective == OBJ_MAKESPAN) {
    for (uint i = 0; i < N; ++i) cost = std::max(cost, D.get(i, C[i]));
  } else if (objective == OBJ_SUM_OF_LOSS) {
    for (uint i = 0; i < N; ++i) cost += D.get(i, C[i]);
  }
  return cost;
}

void Planner::periodic_node_debug(HNode* H, uint loop_count)
{
  const auto now = std::chrono::steady_clock::now();
  if (now - last_debug_print < std::chrono::seconds(4)) return;
  last_debug_print = now;

  const auto config_hash = ConfigHasher{}(H->C);
  std::cout 
            << "depth=" << H->depth
            << " H=" << config_hash
            << " ll=" << H->ll_search
            << " P1=" << ConfigHasher{}(H->parent->C)
            << " ll=" << H->parent->ll_search << "/" << H->parent->max_ll
            << " P2=" << ConfigHasher{}(H->parent->parent->C)
            << " ll=" << H->parent->parent->ll_search << "/" << H->parent->parent->max_ll
            << " i=" << loop_count
            << std::endl;

  // static const std::array<const char*, 10> bucket_labels = {
  //     "0", "1", "2", "3", "4", "5", "6-10", "11-20", "21-50", ">50"};

  // std::cout << "pibt_agents_bucket_counts:";
  // for (size_t i = 0; i < bucket_labels.size(); ++i) {
  //   std::cout << " [" << bucket_labels[i] << "]=" << pibt_agents_bucket_counts[i];
  // }
  // std::cout << std::endl;

  // std::cout << "pibt_cluster_bucket_counts:";
  // for (size_t i = 0; i < bucket_labels.size(); ++i) {
  //   std::cout << " [" << bucket_labels[i] << "]=" << pibt_cluster_bucket_counts[i];
  // }
  // std::cout << std::endl;
}

void Planner::expand_lowlevel_tree(HNode* H, const std::shared_ptr<LNode>& L)
{
  if (L->depth >= N) return;
  const auto i = H->order[L->depth];
  auto C = H->C[i]->neighbor;
  C.push_back(H->C[i]);
  // randomize
  if (MT != nullptr) std::shuffle(C.begin(), C.end(), *MT);
  // insert
  for (auto v : C) {
    auto child = std::make_shared<LNode>(L, i, v);
    H->search_tree.push(child);
  }
}

std::ostream& operator<<(std::ostream& os, const Objective obj)
{
  if (obj == OBJ_NONE) {
    os << "none";
  } else if (obj == OBJ_MAKESPAN) {
    os << "makespan";
  } else if (obj == OBJ_SUM_OF_LOSS) {
    os << "sum_of_loss";
  }
  return os;
}
