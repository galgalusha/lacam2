#include "../include/planner.hpp"

#include "/home/galko/.local/include/CBSH2/ICBSSearch.h"
#include "../include/cbsh2_stuff.hpp"

#include <chrono>
#include <algorithm>

namespace {

size_t pibt_bucket_index(const size_t value)
{
  if (value == 0) return 0;
  if (value == 1) return 1;
  if (value == 2) return 2;
  if (value == 3) return 3;
  if (value == 4) return 4;
  if (value == 5) return 5;
  if (value <= 10) return 6;
  if (value <= 20) return 7;
  if (value <= 50) return 8;
  return 9;
}

}  // namespace

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

bool Planner::wdg_flag = false;
int Planner::max_ll = -1;
float Planner::max_ll_decay = 1.0f;
bool Planner::pibt_clustering = false;
std::array<uint64_t, 10> Planner::pibt_agents_bucket_counts = {0};
std::array<uint64_t, 10> Planner::pibt_cluster_bucket_counts = {0};

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
      C_next(N),
      tie_breakers(V_size, 0),
      A(N, nullptr),
      occupied_now(V_size, nullptr),
      occupied_next(V_size, nullptr),
      last_debug_print(std::chrono::steady_clock::now())
{
}

Planner::~Planner() {}

void Planner::update_pibt_bucket_counters(const HNode* H)
{
  size_t total_agents_in_clusters = 0;
  for (const auto& cluster : H->pibt_cluster) {
    total_agents_in_clusters += cluster.size();
  }

  const auto cluster_count = H->pibt_cluster.size();
  ++pibt_agents_bucket_counts[pibt_bucket_index(total_agents_in_clusters)];
  ++pibt_cluster_bucket_counts[pibt_bucket_index(cluster_count)];
}

void Planner::print_solution_pibt_clusters(const HNode* H_goal) const
{
  if (H_goal == nullptr) return;

  auto chain = std::vector<const HNode*>();
  for (auto p = H_goal; p != nullptr; p = p->parent) {
    chain.push_back(p);
  }
  std::reverse(chain.begin(), chain.end());

  for (size_t step = 1; step < chain.size(); ++step) {
    std::cout << "--- step " << step << " --" << std::endl;
    for (const auto& cluster : chain[step - 1]->pibt_cluster) {
      for (size_t i = 0; i < cluster.size(); ++i) {
        if (i != 0) std::cout << " ";
        std::cout << cluster[i];
      }
      std::cout << std::endl;
    }
  }
}

void Planner::load_cbsh_values()
{
  cbsh_cache.clear();

  std::ifstream in("cbsh_values.txt");
  if (!in.is_open()) return;

  size_t config_hash = 0;
  uint h = 0;
  while (in >> config_hash >> h) {
    cbsh_cache[config_hash] = h;
  }
}

uint Planner::get_or_compute_cbs_heuristic(const Config& C)
{
  const auto config_hash = ConfigHasher{}(C);
  const auto it = cbsh_cache.find(config_hash);
  if (it != cbsh_cache.end()) {
    return it->second;
  }

  auto h = cbs_heuristic(C);

  cbsh_cache[config_hash] = h;
  return h;
}

Solution Planner::solve(std::string& additional_info)
{
  solver_info(1, "start search");

  if (wdg_flag) {
    load_cbsh_values();

    cbsh_values_file.open("cbsh_values.txt", std::ios::out | std::ios::app);
    if (!cbsh_values_file.is_open()) {
      std::cerr << "warning: failed to open cbsh_values.txt for writing" << std::endl;
    }
  }

  // setup agents
  for (auto i = 0; i < N; ++i) A[i] = new Agent(i);

  // setup search
  auto OPEN = std::stack<HNode*>();
  auto EXPLORED = std::unordered_map<Config, HNode*, ConfigHasher>();
  // insert initial node, 'H': high-level node
  auto H_init = new HNode(ins->starts, D, nullptr, 0, get_h_value(ins->starts));
  if (wdg_flag) {
    H_init->h_cbs = get_or_compute_cbs_heuristic(H_init->C);
    H_init->f += H_init->h_cbs;
  }
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
    periodic_node_debug(H, loop_cnt);

    // low-level search end
    if (H->search_tree.empty()) {
      OPEN.pop();
      continue;
    }

    // check lower bounds
    if (H_goal != nullptr && H->f >= H_goal->f) {
//      if (wdg_flag) set_wdg_to_parents(H);
      OPEN.pop();
      continue;
    }

    // check goal condition
    if (H_goal == nullptr && is_same_config(H->C, ins->goals)) {
      H_goal = H;
      solver_info(1, "found solution, cost: ", H->g);
      // set_wdg_to_parents(H);
      auto chain = std::vector<HNode*>();
      auto p = H_goal;
      while (p != nullptr) {
        chain.push_back(p);
        p = p->parent;
      }
      std::reverse(chain.begin(), chain.end());
      std::cout << "h_cbs path:";
      for (auto n : chain) std::cout << " " << n->h_cbs;
      std::cout << std::endl;
      // print_solution_pibt_clusters(H_goal);
      if (objective == OBJ_NONE) break;
      continue;
    }

    if (H->max_ll > -1 && static_cast<float>(H->ll_search) > H->max_ll) {
      if (H->parent != nullptr && !H->parent->max_llalready_decayed) {
        H->parent->max_llalready_decayed = true;
        H->parent->max_ll = std::max(H->max_ll * Planner::max_ll_decay, 2.0f);
      }
      OPEN.pop();
      continue;
    }

    if (H_goal != nullptr && loop_cnt % 5000 == 0) {
      OPEN.push(H_init);
      continue;
    }

    // create successors at the low-level search
    auto L = H->search_tree.front();
    H->search_tree.pop();    
    H->ll_search += 1;

    expand_lowlevel_tree(H, L);

    // create successors at the high-level search
    const auto res = get_new_config(H, L.get());
    if (!res) continue;

    // create new configuration
    for (auto a : A) C_new[a->id] = a->v_next;

    // check explored list
    const auto iter = EXPLORED.find(C_new);
    if (iter != EXPLORED.end()) {
      // case found
      const auto old_goal_cost = (H_goal != nullptr) ? H_goal->g : 0;
      rewrite(H, iter->second, H_goal, OPEN);
      // if (H_goal != nullptr && H_goal->g < old_goal_cost) {
      //   print_solution_pibt_clusters(H_goal);
      // }

      // re-insert or random-restart
      auto H_insert = (MT != nullptr && get_random_float(MT) >= RESTART_RATE)
                          ? iter->second
                          : H_init;
      if (H_goal == nullptr || H_insert->f < H_goal->f) OPEN.push(H_insert);
      // set_wdg_to_parents(H);
    } else {
      // insert new search node
      update_pibt_bucket_counters(H);
      auto new_g = H->g + get_edge_cost(H->C, C_new);
      auto new_h = get_h_value(C_new);
      auto new_f = new_g + new_h;
      if (H_goal != nullptr && new_f >= H_goal->f)
        continue;
      uint h_cbs = wdg_flag ? get_or_compute_cbs_heuristic(C_new) : 0;
      new_f += h_cbs;
      if (H_goal != nullptr && new_f >= H_goal->f)
        continue;
      const auto H_new = new HNode(C_new, D, H, new_g, new_h);
      H_new->h_cbs = h_cbs;
      H_new->f += h_cbs;
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
  for (auto a : A) delete a;
  for (auto itr : EXPLORED) delete itr.second;

  return solution;
}

uint get_depth(HNode* H) {
  uint depth = 0;
  for (auto p = H->parent; p != nullptr; p = p->parent) {
    depth += 1;
  }
  return depth;  
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
      auto g_val = n_from->g + get_edge_cost(n_from->C, n_to->C);
      if (g_val < n_to->g) {
        if (n_to == H_goal)
          solver_info(1, "depth(H): ", get_depth(H_from), ", depth(G): ",get_depth(H_goal), ", cost update: ", n_to->g, " -> ", g_val);
        n_to->g = g_val;
        n_to->f = n_to->g + n_to->h;
        n_to->parent = n_from;
        // n_to->incoming_pibt_clusters = n_from->pibt_clusters;
        Q.push(n_to);
        if (H_goal != nullptr && n_to->f < H_goal->f) OPEN.push(n_to);
      }
    }
  }
}

uint Planner::get_edge_cost(const Config& C1, const Config& C2)
{
  if (objective == OBJ_SUM_OF_LOSS) {
    uint cost = 0;
    for (uint i = 0; i < N; ++i) {
      if (C1[i] != ins->goals[i] || C2[i] != ins->goals[i]) {
        cost += 1;
      }
    }
    return cost;
  }

  // default: makespan
  return 1;
}

uint Planner::get_edge_cost(HNode* H_from, HNode* H_to)
{
  return get_edge_cost(H_from->C, H_to->C);
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
            << "depth=" << get_depth(H) 
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

uint Planner::cbs_heuristic(const Config& C)
{
  load_cbs_agents(*ins, C);
  auto time_limit = 4000;
  auto screen = 0;
  auto search = ICBSSearch(*cbs_map, *cbs_agents, 1.0, WG, true, false, time_limit, screen);
  ICBSNode& start_node = *search.dummy_start;
  uint h = static_cast<uint>(search.computeHeuristics(start_node));
  if (h == -1) {
    std::cout << "cbs_heuristic timed out" << std::endl;
    h = 1000000u;
  }

  if (cbsh_values_file.is_open()) {
    const auto config_hash = ConfigHasher{}(C);
    cbsh_values_file << config_hash << " " << h << "\n";
    cbsh_values_file.flush();
  }

  return h;
}

void Planner::set_wdg_to_parents(HNode* H)
{
  if (H->h_cbs == 0) {
    H->h_cbs = get_or_compute_cbs_heuristic(H->C);
  }

  std::vector<HNode*> missing_parents;
  HNode* found_parent_wdg = nullptr;

  for (auto p = H->parent; p != nullptr; p = p->parent) {
    const auto p_hash = ConfigHasher{}(p->C);
    std::cout << "checking parent " << p_hash;
    const auto it_p = cbsh_cache.find(p_hash);
    if (it_p == cbsh_cache.end()) {
      std::cout << ", Computing missing wdg... ";
      p->h_cbs = get_or_compute_cbs_heuristic(p->C);
      std::cout << p->h_cbs << std::endl;
      // missing_parents.push_back(p);
      continue;
    }
    std::cout << " already has wdg: " << p->h_cbs << ", " << it_p->second << std::endl;
    found_parent_wdg = p;
    break;
  }
}

void Planner::expand_lowlevel_tree(HNode* H, const std::shared_ptr<LNode>& L)
{
  if (L->depth >= N) return;
  const auto i = H->constraint_order[L->depth];
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
