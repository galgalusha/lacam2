#include "../include/planner.hpp"
#include "../include/pibt.hpp"
#include "../include/policy_pibt.hpp"
#include "../include/rollout_result.hpp"

#include <algorithm>
#include <future>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>


const uint GET_ITERATIONS = 1000;
const uint NUM_OF_SUCCESSORS = 80;
const uint NUM_OF_THREADS = 6;


std::vector<WPlanner::Successor> WPlanner::get_successors(
  HNode* H, uint& best_temp_cost, uint64_t& num_node_gen,
    const uint num_expansions, const uint num_of_successors,
    const bool save_rollouts, const PIBTFactory& pibt_factory)
{
  auto successors = std::vector<Successor>();
  if (H == nullptr) return successors;
  if (num_of_successors == 0) return successors;

  auto C_new = Config(N, nullptr);
  auto successor_cmp = [](const Successor& lhs, const Successor& rhs) {
    if (lhs.temp_cost != rhs.temp_cost) return lhs.temp_cost < rhs.temp_cost;
    if (lhs.depth != rhs.depth) return lhs.depth < rhs.depth;
    return lhs.node < rhs.node;
  };
  auto best_successors = std::set<Successor, decltype(successor_cmp)>(
      successor_cmp);

  std::vector<std::mt19937> rollout_rngs(NUM_OF_THREADS);
  if (MT != nullptr) {
    for (uint i = 0; i < NUM_OF_THREADS; ++i) {
      rollout_rngs[i].seed((*MT)());
    }
  } else {
    for (uint i = 0; i < NUM_OF_THREADS; ++i) {
      rollout_rngs[i].seed(i + 1);
    }
  }

  std::vector<std::unique_ptr<PIBTBase>> pibts;
  pibts.reserve(NUM_OF_THREADS);
  for (uint i = 0; i < NUM_OF_THREADS; ++i) {
    if (pibt_factory) {
      pibts.push_back(pibt_factory(&rollout_rngs[i]));
    } else {
      pibts.push_back(std::make_unique<PIBT>(ins, D, &rollout_rngs[i]));
    }
  }

  uint expansions_done = 0;
  while (expansions_done < num_expansions && !is_expired(deadline)) {
    struct BatchItem {
      uint worker_id;
      HNode* node;
      RolloutResult rollout_res;
      bool success;
      bool is_goal;
      uint temp_cost;
    };

    const uint remaining = num_expansions - expansions_done;
    uint batch_size = std::min(remaining, NUM_OF_THREADS);
    if (batch_size == 0) break;

    // No more low-level nodes to expand from this node.
    if (H->search_tree.empty()) break;

    std::vector<BatchItem> batch_items;
    batch_items.reserve(batch_size);

    for (uint worker_id = 0;
         worker_id < batch_size && !H->search_tree.empty() && !is_expired(deadline);
         ++worker_id) {
      loop_cnt += 1;
      expansions_done += 1;

      auto L = H->search_tree.front();
      H->search_tree.pop();
      H->ll_search += 1;

      expand_lowlevel_tree(H, L);

      const auto res = this->pibt->get_new_config(H, L.get(), C_new);
      if (!res) continue;

      auto new_g = H->g + this->pibt->get_edge_cost(H->C, C_new);
      auto H_new = new HNode(C_new, D, H, new_g, 0);
      batch_items.push_back(
          {worker_id, H_new, {false, 0, 0}, false, false, UINT_MAX});
    }

    if (batch_items.empty()) {
      continue;
    }

    std::vector<std::future<RolloutResult>> futures;
    futures.reserve(batch_items.size());
    for (const auto& item : batch_items) {
      const uint worker_id = item.worker_id;
      HNode* node = item.node;
      futures.push_back(std::async(std::launch::async, [&pibts, worker_id, node]() {
        return pibts[worker_id]->rollout(node);
      }));
    }

    for (size_t i = 0; i < batch_items.size(); ++i) {
      batch_items[i].rollout_res = futures[i].get();
      if (!batch_items[i].rollout_res.success) {
        delete batch_items[i].node;
        continue;
      }

      auto* H_new = batch_items[i].node;
      H_new->h = batch_items[i].rollout_res.cost;
      H_new->f = H_new->g + H_new->h;

      num_node_gen += 1;
      batch_items[i].success = true;
      batch_items[i].is_goal = is_same_config(H_new->C, ins->goals);
      batch_items[i].temp_cost = H_new->g + batch_items[i].rollout_res.cost;

      if (batch_items[i].temp_cost < best_temp_cost) {
        best_temp_cost = batch_items[i].temp_cost;
      }
    }

    BatchItem* best_goal_item = nullptr;
    for (auto& item : batch_items) {
      if (!item.success || !item.is_goal) continue;
      if (best_goal_item == nullptr || item.node->g < best_goal_item->node->g) {
        best_goal_item = &item;
      }
    }

    if (best_goal_item != nullptr) {
      std::cout << "goal found" << std::endl;
      const uint goal_temp_cost = best_goal_item->node->g;
      if (goal_temp_cost < best_temp_cost) {
        best_temp_cost = goal_temp_cost;
      }

      for (const auto& successor : best_successors) {
        delete successor.node;
      }

      successors.clear();
      for (auto& item : batch_items) {
        if (&item != best_goal_item && item.success) {
          delete item.node;
        }
      }

      successors.push_back({best_goal_item->node,
                            static_cast<uint>(best_goal_item->node->depth),
                            goal_temp_cost,
                            save_rollouts ? best_goal_item->rollout_res.configs : std::vector<Config>{}});
      return successors;
    }

    for (const auto& item : batch_items) {
      if (!item.success) continue;
      Successor candidate{
          item.node, static_cast<uint>(item.node->depth), item.temp_cost,
          save_rollouts ? item.rollout_res.configs : std::vector<Config>{}};
      best_successors.insert(candidate);
    }
  }

  while (best_successors.size() > num_of_successors) {
    auto worst_it = std::prev(best_successors.end());
    delete worst_it->node;
    best_successors.erase(worst_it);
  }

  successors.reserve(best_successors.size());
  for (const auto& successor : best_successors) {
    successors.push_back(successor);
  }

  return successors;
}

NeighborScorePolicy WPlanner::create_policy(int num_agents) {
  return create_policy(ins->starts, num_agents);
}

NeighborScorePolicy WPlanner::create_policy(const Config& start_config, int num_agents) {
  auto H_init = new HNode(start_config, D, nullptr, 0, 0);
  uint64_t node_counter = 0;
  uint best_cost = UINT_MAX;
  const uint n_expansions = 3000;
  const uint n_rollouts = 100;
  std::cout << "generating " << n_expansions << " rollouts for policy" << std::endl;
  auto best_rollouts = get_successors(H_init, best_cost, node_counter, n_expansions, n_rollouts, true);
  std::cout << "Done generating " << n_expansions << " rollouts for policy. Best cost found: " << best_cost << std::endl;

  std::vector<AgentPolicy> agent_policies(num_agents);

  for (const auto& successor : best_rollouts) {
    const auto& rollout = successor.rollout;
    for (size_t step = 1; step < rollout.size(); ++step) {
      const Config& prev = rollout[step - 1];
      const Config& curr = rollout[step];
      for (int agent = 0; agent < num_agents; ++agent) {
        Vertex* from = prev[agent];
        Vertex* to = curr[agent];
        if (from != to) {
          agent_policies[agent].record_move(from, to);
        }
      }
    }
  }

  delete H_init;
  return NeighborScorePolicy(std::move(agent_policies), MT);
}

void WPlanner::test_policy(int TEST_AGENT_ID)
{
  auto policy = create_policy(ins->N);

  const uint W = ins->G.width;
  const uint H = ins->G.height;
  Vertex* start = ins->starts[TEST_AGENT_ID];
  Vertex* goal  = ins->goals[TEST_AGENT_ID];

  const Config& C = ins->starts;

  std::vector<std::string> grid(H, std::string(W, '#'));

  for (auto* v : ins->G.V) {
    uint x = v->index % W;
    uint y = v->index / W;
    grid[y][x] = '.';
  }

  for (auto* v : ins->G.V) {
    uint x = v->index % W;
    uint y = v->index / W;

    if (v == start) { grid[y][x] = 'S'; continue; }
    if (v == goal)  { grid[y][x] = 'G'; continue; }

    if (v->neighbor.empty()) continue;

    Vertices neighbors(v->neighbor.begin(), v->neighbor.end());
    neighbors.push_back(v);

    Config C_tmp = C;
    C_tmp[TEST_AGENT_ID] = v;

    auto scores = policy.get_neighbor_scores(C_tmp, TEST_AGENT_ID, neighbors);

    Vertex* best = nullptr;
    float best_score = 1.0f;
    for (auto* nb : v->neighbor) {
      float s = scores.count(nb) ? scores.at(nb) : 0.0f;
      if (s < best_score) { best_score = s; best = nb; }
    }

    if (best == nullptr) continue;

    uint bx = best->index % W;
    uint by = best->index / W;
    char dir = '.';
    if      (bx < x) dir = 'L';
    else if (bx > x) dir = 'R';
    else if (by < y) dir = 'U';
    else if (by > y) dir = 'D';

    grid[y][x] = dir;
  }

  std::cout << "=== Policy map for agent " << TEST_AGENT_ID << " ===" << std::endl;
  for (const auto& row : grid) std::cout << row << "\n";
  std::cout << "==================" << std::endl;
}

Solution WPlanner::solve(std::string& additional_info)
{
  const uint refresh_policy_depth = 600;

  auto policy = std::make_shared<NeighborScorePolicy>(create_policy(ins->N));
  PIBTFactory policy_pibt_factory = [&](std::mt19937* rng) -> std::unique_ptr<PIBTBase> {
    return std::make_unique<PolicyPIBT>(ins, D, policy);
  };

  auto cmp = [](const HNode* lhs, const HNode* rhs) {
    auto l = lhs->g + lhs->h * 1.05;
    auto r = rhs->g + rhs->h * 1.05;
    if (l != r) return l > r;

    //if (lhs->f != rhs->f) return lhs->f > rhs->f;
    return lhs->depth > rhs->depth;
  };
  auto frontier = std::priority_queue<HNode*, std::vector<HNode*>, decltype(cmp)>(cmp);
  auto best_seen_g = std::unordered_map<Config, uint, ConfigHasher>();
  std::vector<Config> solution;
  auto H_init = new HNode(ins->starts, D, nullptr, 0, 0);
  HNode* H_goal = nullptr;
  std::vector<HNode*> all_nodes;
  all_nodes.push_back(H_init);
  frontier.push(H_init);
  best_seen_g[H_init->C] = H_init->g;

  uint best_goal_cost = UINT_MAX;
  uint best_temp_cost = UINT_MAX;
  uint64_t num_node_gen = 1;
  auto last_print_time = std::chrono::steady_clock::now();
  uint last_printed_best_temp = UINT_MAX;

  // Track best Config (lowest g) seen at each depth
  std::unordered_map<uint, std::pair<Config, uint>> best_config_at_depth;
  // Track which depths have been first expanded
  std::unordered_set<uint> expanded_depths;

  // Record initial node
  best_config_at_depth[0] = {H_init->C, H_init->g};

  while (!frontier.empty() && !is_expired(deadline)) {
    auto* H = frontier.top();
    frontier.pop();

    // Branch-and-bound pruning: edge costs are non-negative.
    if (best_goal_cost != UINT_MAX && H->g >= best_goal_cost) {
      continue;
    }

    // Check if this depth is being expanded for the first time
    const uint cur_depth = static_cast<uint>(H->depth);
    if (expanded_depths.find(cur_depth) == expanded_depths.end()) {
      expanded_depths.insert(cur_depth);
      if (cur_depth % refresh_policy_depth == 0 && cur_depth >= refresh_policy_depth * 2) {
        const uint ref_depth = cur_depth / 2;
        auto it = best_config_at_depth.find(ref_depth);
        if (it != best_config_at_depth.end()) {
          std::cout << "WPlanner: refreshing policy at depth=" << cur_depth
                    << " using best config from depth=" << ref_depth << std::endl;
            policy = std::make_shared<NeighborScorePolicy>(create_policy(ins->N));
            policy_pibt_factory = [&](std::mt19937* rng) -> std::unique_ptr<PIBTBase> {
              return std::make_unique<PolicyPIBT>(ins, D, policy);
            };


        } else {
          std::cout << "WPlanner: could not refresh policy at depth=" << cur_depth << std::endl;
        }
      }
    }

    auto get_iterations = GET_ITERATIONS / (sqrt(H->depth + 1));
    auto num_of_successors = NUM_OF_SUCCESSORS / (sqrt(H->depth + 1));
    auto next_nodes =
      get_successors(H, best_temp_cost, num_node_gen, get_iterations,
               NUM_OF_SUCCESSORS, false, policy_pibt_factory);

    HNode* best_goal_successor = nullptr;
    for (const auto& successor : next_nodes) {
      auto* H_next = successor.node;
      if (!is_same_config(H_next->C, ins->goals)) continue;
      if (best_goal_successor == nullptr || H_next->g < best_goal_successor->g) {
        best_goal_successor = H_next;
      }
    }

    if (best_goal_successor != nullptr) {
      for (const auto& successor : next_nodes) {
        auto* H_next = successor.node;
        if (H_next != best_goal_successor) {
          delete H_next;
        }
      }

      all_nodes.push_back(best_goal_successor);
      if (H_goal == nullptr || best_goal_successor->g < H_goal->g) {
        H_goal = best_goal_successor;
      }
      if (best_goal_successor->g < best_goal_cost) {
        best_goal_cost = best_goal_successor->g;
      }

      std::cout << "WPlanner goal successor: best_goal_cost="
                << (best_goal_cost == UINT_MAX ? -1 : static_cast<int>(best_goal_cost))
                << " best_temp_cost="
                << (best_temp_cost == UINT_MAX ? -1 : static_cast<int>(best_temp_cost))
                << " depth=" << best_goal_successor->depth
                << " f=" << best_goal_successor->f << std::endl;
      continue;
    }

    for (const auto& successor : next_nodes) {
      auto* H_next = successor.node;
      if (best_goal_cost != UINT_MAX && H_next->g >= best_goal_cost) {
        delete H_next;
        continue;
      }
      const auto seen_it = best_seen_g.find(H_next->C);
      if (seen_it != best_seen_g.end() && seen_it->second <= H_next->g) {
        delete H_next;
        continue;
      }
      best_seen_g[H_next->C] = H_next->g;

      // Record best config for this depth
      const uint next_depth = static_cast<uint>(H_next->depth);
      auto depth_it = best_config_at_depth.find(next_depth);
      if (depth_it == best_config_at_depth.end() || H_next->g < depth_it->second.second) {
        best_config_at_depth[next_depth] = {H_next->C, H_next->g};
      }

      frontier.push(H_next);
      all_nodes.push_back(H_next);

      auto now = std::chrono::steady_clock::now();
      bool temp_updated = best_temp_cost != last_printed_best_temp;
      bool timeout_print = std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count() >= 2;
      if (temp_updated || timeout_print) {
        last_print_time = now;
        last_printed_best_temp = best_temp_cost;
        std::cout << "WPlanner successor: best_goal_cost="
            << (best_goal_cost == UINT_MAX ? -1 : static_cast<int>(best_goal_cost))
            << " best_temp_cost="
            << (best_temp_cost == UINT_MAX ? -1 : static_cast<int>(best_temp_cost))
                  << " depth=" << H_next->depth
                  << " f=" << H_next->f
                  << " successor_depth=" << successor.depth
                  << " temp_cost=" << successor.temp_cost
                  << std::endl;
      }
    }
  }

  if (H_goal != nullptr) {
    auto H = H_goal;
    while (H != nullptr) {
      solution.push_back(H->C);
      H = H->parent;
    }
    std::reverse(solution.begin(), solution.end());
  }

  if (H_goal != nullptr) {
    solver_info(1, "found solution, depth: ", H_goal->depth, ", g: ", H_goal->g);
  } else if (is_expired(deadline)) {
    solver_info(1, "timeout");
  } else {
    solver_info(1, "search exhausted");
  }

  additional_info += "planner=WPlanner\n";
  additional_info += "objective=" + std::to_string(objective) + "\n";
  additional_info += "loop_cnt=" + std::to_string(loop_cnt) + "\n";
  additional_info += "num_node_gen=" + std::to_string(num_node_gen) + "\n";
  additional_info +=
      "best_cost=" +
      std::to_string(best_goal_cost == UINT_MAX ? -1 : (int)best_goal_cost) +
      "\n";
    additional_info +=
      "best_temp_cost=" +
      std::to_string(best_temp_cost == UINT_MAX ? -1 : (int)best_temp_cost) +
      "\n";

  for (auto node : all_nodes) delete node;

  return solution;
}
