#pragma once

#include "/home/galko/.local/include/CBSH2/map_loader.h"
#include "/home/galko/.local/include/CBSH2/agents_loader.h"
#include "/home/galko/.local/include/CBSH2/ICBSNode.h"
#include "instance.hpp"


inline constexpr heuristics_type WG = WDG;


inline MapLoader* cbs_map = nullptr;

inline AgentsLoader* cbs_agents = nullptr;


inline void load_cbs_agents(const Instance& ins, const Config& starts) {
  if (cbs_agents == nullptr) cbs_agents = new AgentsLoader();
  cbs_agents->num_of_agents = 0;
  cbs_agents->goal_locations.clear();
  cbs_agents->initial_locations.clear();
  uint W = ins.G.width;
  for (uint i = 0; i < ins.N; i++) {
      auto start_y = starts[i]->index / W;
      auto start_x = starts[i]->index % W;
      auto goal_y = ins.goals[i]->index / W;
      auto goal_x = ins.goals[i]->index % W;
      cbs_agents->addAgent(start_y, start_x, goal_y, goal_x);
  }
}

inline void load_cbs_agents(const Instance& ins) {
  load_cbs_agents(ins, ins.starts);
}

