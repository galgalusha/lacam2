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

  const uint N;
  const uint V_size;
  DistTable D;

  ClusteredPlanner(const Instance* _ins, const Deadline* _deadline,
                   std::mt19937* _MT, const int _verbose = 0);

  Solution solve();
};
