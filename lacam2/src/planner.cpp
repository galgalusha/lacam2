#include "../include/planner.hpp"

#include "/home/galko/.local/include/CBSH2/ICBSSearch.h"
#include "../include/cbsh2_stuff.hpp"

#include <chrono>

LNode::LNode(LNode* parent, uint i, Vertex* v)
    : who(), where(), depth(parent == nullptr ? 0 : parent->depth + 1)
{
  if (parent != nullptr) {
    who = parent->who;
    who.push_back(i);
    where = parent->where;
    where.push_back(v);
  }
}

uint HNode::HNODE_CNT = 0;
bool Planner::wdg_flag = false;
int Planner::max_ll_depth = -1;

// for high-level
HNode::HNode(const Config& _C, DistTable& D, HNode* _parent, const uint _g,
             const uint _h)
    : C(_C),
      parent(_parent),
      neighbor(),
      g(_g),
      h(_h),
      h_cbs(0),
      f(g + h),
      priorities(C.size()),
      order(C.size(), 0),
      search_tree(std::queue<LNode*>()),
      ll_search(0)
{
  ++HNODE_CNT;

  search_tree.push(new LNode());
  ll_search += 1;
  const auto N = C.size();

  // update neighbor
  if (parent != nullptr) parent->neighbor.insert(this);

  // set priorities
  if (parent == nullptr) {
    // initialize
    for (uint i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
  } else {
    // dynamic priorities, akin to PIBT
    for (size_t i = 0; i < N; ++i) {
      if (D.get(i, C[i]) != 0) {
        priorities[i] = parent->priorities[i] + 1;
      } else {
        priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
      }
    }
  }

  // set order
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](uint i, uint j) { return priorities[i] > priorities[j]; });
}

HNode::~HNode()
{
  while (!search_tree.empty()) {
    delete search_tree.front();
    search_tree.pop();
  }
}

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

uint Planner::get_or_compute_cbs_heuristic(HNode* H)
{
  const auto config_hash = ConfigHasher{}(H->C);
  const auto it = cbsh_cache.find(config_hash);
  if (it != cbsh_cache.end()) {
    return it->second;
  }

  const auto h = cbs_heuristic(H);
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
    H_init->h_cbs = get_or_compute_cbs_heuristic(H_init);
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
    periodic_node_debug(H);

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
      if (objective == OBJ_NONE) break;
      continue;
    }

    // create successors at the low-level search
    auto L = H->search_tree.front();
    H->search_tree.pop();
    H->ll_search += 1;
    if (max_ll_depth >= 0 && static_cast<int>(L->depth) > max_ll_depth) {
      delete L;
      OPEN.pop();
      continue;
    }
    expand_lowlevel_tree(H, L);

    // create successors at the high-level search
    const auto res = get_new_config(H, L);
    delete L;  // free
    if (!res) continue;

    // create new configuration
    for (auto a : A) C_new[a->id] = a->v_next;

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
      // set_wdg_to_parents(H);
    } else {
      // insert new search node
      const auto H_new = new HNode(
          C_new, D, H, H->g + get_edge_cost(H->C, C_new), get_h_value(C_new));
      if (wdg_flag) { // loop_cnt % 100 == 0) {
        H_new->h_cbs = get_or_compute_cbs_heuristic(H_new);
        H_new->f += H_new->h_cbs;
      }
      EXPLORED[H_new->C] = H_new;
      if (H_goal == nullptr || H_new->f < H_goal->f) OPEN.push(H_new);
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
          solver_info(1, "cost update: ", n_to->g, " -> ", g_val);
        n_to->g = g_val;
        n_to->f = n_to->g + n_to->h;
        n_to->parent = n_from;
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

void Planner::periodic_node_debug(HNode* H)
{
  const auto now = std::chrono::steady_clock::now();
  if (now - last_debug_print < std::chrono::seconds(10)) return;
  last_debug_print = now;

  uint depth = 0;
  for (auto p = H->parent; p != nullptr; p = p->parent) {
    depth += 1;
  }

  const auto config_hash = ConfigHasher{}(H->C);
  std::cout << "hash=" << config_hash
            << " ll_search=" << H->ll_search
            << " depth=" << depth << std::endl;
}

uint Planner::cbs_heuristic(HNode* H)
{
  load_cbs_agents(*ins, H->C);
  auto time_limit = 500;
  auto screen = 0;
  auto search = ICBSSearch(*cbs_map, *cbs_agents, 1.0, WG, true, false, time_limit, screen);
  ICBSNode& start_node = *search.dummy_start;
  uint h = static_cast<uint>(search.computeHeuristics(start_node));

  if (cbsh_values_file.is_open()) {
    const auto config_hash = ConfigHasher{}(H->C);
    cbsh_values_file << config_hash << " " << h << "\n";
    cbsh_values_file.flush();
  }

  return h;
}

void Planner::set_wdg_to_parents(HNode* H)
{
  if (H->h_cbs == 0) {
    H->h_cbs = get_or_compute_cbs_heuristic(H);
  }

  std::vector<HNode*> missing_parents;
  HNode* found_parent_wdg = nullptr;

  for (auto p = H->parent; p != nullptr; p = p->parent) {
    const auto p_hash = ConfigHasher{}(p->C);
    std::cout << "checking parent " << p_hash;
    const auto it_p = cbsh_cache.find(p_hash);
    if (it_p == cbsh_cache.end()) {
      std::cout << ", Computing missing wdg... ";
      p->h_cbs = get_or_compute_cbs_heuristic(p);
      std::cout << p->h_cbs << std::endl;
      // missing_parents.push_back(p);
      continue;
    }
    std::cout << " already has wdg: " << p->h_cbs << ", " << it_p->second << std::endl;
    found_parent_wdg = p;
    break;
  }

  // // if (!found_parent_wdg) return;
  // // auto min_wdg = std::min(H->h_cbs, found_parent_wdg->h_cbs);

  // // for (auto p : missing_parents) {
  // //   const auto p_hash = ConfigHasher{}(p->C);
  // //   cbsh_cache[p_hash] = min_wdg;
  // //   if (p->h_cbs == 0) {
  // //     p->h_cbs = min_wdg;
  // //     p->f += min_wdg;
  // //   }
  // // }
  // if (missing_parents.size()) {
  //   std::cout << "H cbs " << H->h_cbs << " found parent: " << found_parent_wdg->h_cbs << std::endl;
  //   std::cout << "Updating " << missing_parents.size() << " parents to h_cbs=" << min_wdg << std::endl;
  // }
}

void Planner::expand_lowlevel_tree(HNode* H, LNode* L)
{
  if (L->depth >= N) return;
  const auto i = H->order[L->depth];
  auto C = H->C[i]->neighbor;
  C.push_back(H->C[i]);
  // randomize
  if (MT != nullptr) std::shuffle(C.begin(), C.end(), *MT);
  // insert
  for (auto v : C) {
    H->search_tree.push(new LNode(L, i, v));
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
