#pragma once

#include "/home/galko/.local/include/CBSH2/map_loader.h"
#include "/home/galko/.local/include/CBSH2/agents_loader.h"
#include "/home/galko/.local/include/CBSH2/ICBSNode.h"
#include "instance.hpp"


inline constexpr heuristics_type WG = WDG;


inline MapLoader* cbs_map = nullptr;

inline AgentsLoader* cbs_agents = nullptr;

inline ICBSNode cbs_dummy_node;
inline const bool cbs_dummy_node_initialized = []() {
  cbs_dummy_node.parent = nullptr;
  cbs_dummy_node.agent_id = -1;
  cbs_dummy_node.constraints.clear();
  cbs_dummy_node.conflictGraph.clear();
  cbs_dummy_node.rectSemiConf.clear();
  cbs_dummy_node.rectNonConf.clear();
  cbs_dummy_node.cardinalConf.clear();
  cbs_dummy_node.semiConf.clear();
  cbs_dummy_node.nonConf.clear();
  cbs_dummy_node.unknownConf.clear();
  cbs_dummy_node.conflict.reset();
  cbs_dummy_node.path.clear();
  cbs_dummy_node.g_val = 0;
  cbs_dummy_node.h_val = 0;
  cbs_dummy_node.f_val = 0;
  cbs_dummy_node.depth = 0;
  cbs_dummy_node.makespan = 0;
  cbs_dummy_node.num_of_collisions = 0;
  cbs_dummy_node.time_expanded = 0;
  cbs_dummy_node.time_generated = 0;
  return true;
}();

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

