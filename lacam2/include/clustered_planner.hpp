#pragma once

#include "cluster_pibt.hpp"
#include "dist_table.hpp"
#include "instance.hpp"
#include "planner.hpp"
#include "post_processing.hpp"
#include "utils.hpp"

class ClusteredPlanner {
 public:
  const Instance* ins;
  const Deadline* deadline;
  std::mt19937* MT;
  const int verbose;
  const Objective objective;
  const float restart_rate;
  const uint time_window;

  const uint N;
  const uint V_size;
  DistTable D;

  ClusteredPlanner(const Instance* _ins, const Deadline* _deadline,
                   std::mt19937* _MT, const int _verbose = 0,
                   const Objective _objective = OBJ_NONE,
                   const float _restart_rate = 0.001,
                   uint _time_window = 1);

  Solution solve();
};
