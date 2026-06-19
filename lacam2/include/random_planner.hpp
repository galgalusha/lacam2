/*
 * RandomPlanner: random PIBT rollouts from the initial state.
 * Keeps the best (lowest cost) solution found and prints it whenever it
 * improves.  Does not perform a search and does not inherit from Planner.
 */
#pragma once

#include "dist_table.hpp"
#include "instance.hpp"
#include "pibt.hpp"
#include "utils.hpp"

class RandomPlanner {
 public:
  RandomPlanner(const Instance* _ins, const Deadline* _deadline,
                std::mt19937* _MT, int _verbose = 0);

  // Run rollouts until the deadline expires.  Returns the best solution found,
  // or an empty Solution if none succeeded.
  Solution solve();

 private:
  const Instance* ins;
  const Deadline* deadline;
  std::mt19937* MT;
  const int verbose;

  DistTable D;
};
