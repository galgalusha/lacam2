#include "../include/planner.hpp"
#include "../include/pibt.hpp"

#include <algorithm>

Solution WPlanner::solve(std::string& additional_info)
{
  solver_info(1, "start WPlanner search");

  auto OPEN = std::stack<HNode*>();
  auto EXPLORED = std::unordered_map<Config, HNode*, ConfigHasher>();
  auto H_init = new HNode(ins->starts, D, nullptr, 0, get_h_value(ins->starts));
  OPEN.push(H_init);
  EXPLORED[H_init->C] = H_init;

  std::vector<Config> solution;
  auto C_new = Config(N, nullptr);
  uint best_cost = UINT_MAX;

  while (!OPEN.empty() && !is_expired(deadline)) {
    loop_cnt += 1;

    auto H = OPEN.top();
    if (static_cast<size_t>(H->depth) >= depth_visit_counts.size()) {
      depth_visit_counts.resize(H->depth + 1, 0);
    }
    depth_visit_counts[H->depth] += 1;
    periodic_node_debug(H, loop_cnt);

    if (H->depth >= 7) {
      const int rollout_cost = pibt->rollout(H);
      if (rollout_cost >= 0) {
        const uint temp_cost = H->g + static_cast<uint>(rollout_cost);
        if (temp_cost < best_cost) {
          best_cost = temp_cost;
          std::cout << "WPlanner best_cost update: " << best_cost
                    << " (depth=" << H->depth << ", g=" << H->g
                    << ", rollout=" << rollout_cost << ", loop=" << loop_cnt
                    << ")" << std::endl;
        }
      }

      OPEN.pop();
      OPEN.push(H_init);
      continue;
    }

    if (H->search_tree.empty()) {
      OPEN.pop();
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

    auto L = H->search_tree.front();
    H->search_tree.pop();
    H->ll_search += 1;

    expand_lowlevel_tree(H, L);

    const auto res = pibt->get_new_config(H, L.get(), C_new);
    if (!res) continue;

    const auto iter = EXPLORED.find(C_new);
    if (iter != EXPLORED.end()) {
      rewrite(H, iter->second, nullptr, OPEN);

      auto H_insert = (MT != nullptr && get_random_float(MT) >= RESTART_RATE)
                          ? iter->second
                          : H_init;
      OPEN.push(H_insert);
    } else {
      auto new_g = H->g + pibt->get_edge_cost(H->C, C_new);
      auto new_h = get_h_value(C_new);
      const auto H_new = new HNode(C_new, D, H, new_g, new_h);
      H->neighbor.insert(H_new);
      EXPLORED[H_new->C] = H_new;
      OPEN.push(H_new);
    }
  }

  if (OPEN.empty()) {
    solver_info(1, "search exhausted");
  } else {
    solver_info(1, "timeout");
  }

  additional_info += "planner=WPlanner\n";
  additional_info += "objective=" + std::to_string(objective) + "\n";
  additional_info += "loop_cnt=" + std::to_string(loop_cnt) + "\n";
  additional_info += "num_node_gen=" + std::to_string(EXPLORED.size()) + "\n";
  additional_info +=
      "best_cost=" + std::to_string(best_cost == UINT_MAX ? -1 : (int)best_cost) +
      "\n";

  for (auto itr : EXPLORED) delete itr.second;

  return solution;
}
