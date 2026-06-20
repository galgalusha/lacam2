#include "../include/lacam2.hpp"
#include "../include/policy.hpp"
#include "../include/policy_pibt.hpp"



Solution solve(const Instance& ins, std::string& additional_info,
               const int verbose, const Deadline* deadline, std::mt19937* MT,
               const Objective objective, const float restart_rate)
{
  auto planner = Planner(&ins, deadline, MT, verbose, objective, restart_rate,
                         std::make_shared<RandomPolicy>(MT));


  return planner.solve(additional_info);
}

Solution solve_w(const Instance& ins, std::string& additional_info,
                 const int verbose, const Deadline* deadline, std::mt19937* MT,
                 const Objective objective, const float restart_rate)
{
  auto planner = WPlanner(&ins, deadline, MT, verbose, objective, restart_rate);
  return planner.solve(additional_info);
}
