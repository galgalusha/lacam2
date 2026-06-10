#include "../include/planner.hpp"
#include "../include/pibt.hpp"

#include <algorithm>
#include <stack>
#include <unordered_map>
#include <unordered_set>


const int BUCKET_SIZE = 5;

std::vector<WPlanner::BucketedSuccessor> WPlanner::get_successors(
  HNode* H, uint& best_temp_cost, uint64_t& num_node_gen,
    const uint num_expansions)
{
  auto successors = std::vector<BucketedSuccessor>();
  if (H == nullptr) return successors;

  auto C_new = Config(N, nullptr);
  auto best_per_bucket = std::unordered_map<uint, std::pair<uint, HNode*>>();

  for (uint i = 0; i < num_expansions && !is_expired(deadline); ++i) {
    loop_cnt += 1;

    // No more low-level nodes to expand from this node.
    if (H->search_tree.empty()) break;

    auto L = H->search_tree.front();
    H->search_tree.pop();
    H->ll_search += 1;

    expand_lowlevel_tree(H, L);

    const auto res = pibt->get_new_config(H, L.get(), C_new);
    if (!res) continue;

    auto new_g = H->g + pibt->get_edge_cost(H->C, C_new);
    auto new_h = get_h_value(C_new);
    auto H_new = new HNode(C_new, D, H, new_g, new_h);
    num_node_gen += 1;

    const auto rollout_res = pibt->rollout(H_new);
    if (!rollout_res.success) {
      delete H_new;
      continue;
    }

    if (is_same_config(H_new->C, ins->goals)) {
      const uint goal_temp_cost = H_new->g;
      const uint goal_bucket = H_new->depth / BUCKET_SIZE;
      if (goal_temp_cost < best_temp_cost) {
        best_temp_cost = goal_temp_cost;
      }

      for (auto& it : best_per_bucket) {
        delete it.second.second;
      }

      successors.clear();
      successors.push_back({H_new, goal_bucket, goal_temp_cost});
      return successors;
    }

    const uint temp_cost = H_new->g + rollout_res.cost;
    const uint bucket = (rollout_res.makespan + H_new->depth) / BUCKET_SIZE;

    if (temp_cost < best_temp_cost) {
      best_temp_cost = temp_cost;
    }

    const auto bucket_it = best_per_bucket.find(bucket);
    if (bucket_it == best_per_bucket.end()) {
      best_per_bucket.emplace(bucket, std::make_pair(temp_cost, H_new));
    } else if (temp_cost < bucket_it->second.first) {
      delete bucket_it->second.second;
      bucket_it->second = std::make_pair(temp_cost, H_new);
    } else {
      delete H_new;
    }
  }

  successors.reserve(best_per_bucket.size());
  for (auto& it : best_per_bucket) {
    successors.push_back({it.second.second, it.first, it.second.first});
  }

  std::sort(successors.begin(), successors.end(), [](const BucketedSuccessor& lhs,
                                                     const BucketedSuccessor& rhs) {
    if (lhs.temp_cost != rhs.temp_cost) return lhs.temp_cost < rhs.temp_cost;
    return lhs.node->depth < rhs.node->depth;
  });

  return successors;
}


Solution WPlanner::solve(std::string& additional_info)
{
  solver_info(1, "start WPlanner search");

  auto frontier = std::stack<HNode*>();
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

  const uint gen_iterations = 100;

  while (!frontier.empty() && !is_expired(deadline)) {
    auto H = frontier.top();
    frontier.pop();

    // Branch-and-bound pruning: edge costs are non-negative.
    if (best_goal_cost != UINT_MAX && H->g >= best_goal_cost) {
      continue;
    }

    if (is_same_config(H->C, ins->goals)) {
      if (H_goal == nullptr || H->g < H_goal->g) {
        H_goal = H;
      }
      if (H->g < best_goal_cost) {
        best_goal_cost = H->g;
      }
      continue;
    }

    auto next_nodes =
        get_successors(H, best_temp_cost, num_node_gen, gen_iterations);
    std::cout << "--------------------" << std::endl;
    for (auto it = next_nodes.rbegin(); it != next_nodes.rend(); ++it) {
      const auto& successor = *it;
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
                  << " makespan_bucket=" << successor.bucket * BUCKET_SIZE
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
