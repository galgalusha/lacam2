#include "../include/planner.hpp"
#include "../include/pibt.hpp"

#include <algorithm>
#include <future>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>


const uint GET_ITERATIONS = 800;
const uint NUM_OF_SUCCESSORS = 80;
const uint NUM_OF_THREADS = 6;


std::vector<WPlanner::Successor> WPlanner::get_successors(
  HNode* H, uint& best_temp_cost, uint64_t& num_node_gen,
    const uint num_expansions, const uint num_of_successors)
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

  std::vector<std::unique_ptr<PIBT>> pibts;
  pibts.reserve(NUM_OF_THREADS);
  for (uint i = 0; i < NUM_OF_THREADS; ++i) {
    pibts.push_back(std::make_unique<PIBT>(ins, D, &rollout_rngs[i]));
  }

  uint expansions_done = 0;
  while (expansions_done < num_expansions && !is_expired(deadline)) {
    struct BatchItem {
      uint worker_id;
      HNode* node;
      PIBT::RolloutResult rollout_res;
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

      const auto res = pibts[worker_id]->get_new_config(H, L.get(), C_new);
      if (!res) continue;

      auto new_g = H->g + pibts[worker_id]->get_edge_cost(H->C, C_new);
      auto H_new = new HNode(C_new, D, H, new_g, 0);
      batch_items.push_back(
          {worker_id, H_new, {false, 0, 0}, false, false, UINT_MAX});
    }

    if (batch_items.empty()) {
      continue;
    }

    std::vector<std::future<PIBT::RolloutResult>> futures;
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
                            goal_temp_cost});
      return successors;
    }

    for (const auto& item : batch_items) {
      if (!item.success) continue;
      Successor candidate{
          item.node, static_cast<uint>(item.node->depth), item.temp_cost};
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


Solution WPlanner::solve(std::string& additional_info)
{
  solver_info(1, "start WPlanner search");

  auto cmp = [](const HNode* lhs, const HNode* rhs) {
    float f_lhs = lhs->g + lhs->h * 0.95;
    float f_rhs = rhs->g + rhs->h * 0.95;
    if (f_lhs != f_rhs) return f_lhs > f_rhs;
    return lhs->depth > rhs->depth;
    // if (lhs->f != rhs->f) return lhs->f > rhs->f;
    // return lhs->depth > rhs->depth;
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

  while (!frontier.empty() && !is_expired(deadline)) {
    auto* H = frontier.top();
    frontier.pop();

    // Branch-and-bound pruning: edge costs are non-negative.
    if (best_goal_cost != UINT_MAX && H->g >= best_goal_cost) {
      continue;
    }

    auto get_iterations = GET_ITERATIONS / (sqrt(H->depth + 1));
    auto num_of_successors = NUM_OF_SUCCESSORS / (sqrt(H->depth + 1));
    auto next_nodes =
      get_successors(H, best_temp_cost, num_node_gen, get_iterations,
               NUM_OF_SUCCESSORS);

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

      {
        frontier.push(H_next);
        all_nodes.push_back(H_next);
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
