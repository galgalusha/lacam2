#include <argparse/argparse.hpp>
#include <lacam2.hpp>
#include "gui.hpp"
#include <odplanner.hpp>
#include <random_planner.hpp>
#include <clustered_planner.hpp>
#include <clustered_hnode.hpp>
#include <algorithm>
#include <cassert>
#include <cluster_detection_pibt.hpp>
#include <cem_planner.hpp>

static void test_clusters()
{
  // Use a real ClusterDetectionPIBT instance (time_window=2, 6 agents).
  // Map/instance are only needed to satisfy the constructor; the actual
  // cluster logic only depends on recorded interactions and N.
  std::mt19937 MT(0);
  const Instance ins("./assets/random-11-5.map", &MT, 6);
  DistTable D(&ins);
  ClusterDetectionPIBT cpibt(&ins, D, &MT, /*time_window=*/2);

  // Record interactions at explicit timesteps:
  //   window 0 (t=0..1): (1,3)@0, (2,3)@1  → cluster {1,2,3}
  //   window 1 (t=2..3): (2,4)@2, (1,5)@3  → clusters {2,4} and {1,5}
  cpibt.record_interaction(1, 3, 0);
  cpibt.record_interaction(2, 3, 1);
  cpibt.record_interaction(2, 4, 2);
  cpibt.record_interaction(1, 5, 3);

  cpibt.compute_clusters();

  auto print_clusters = [](const std::vector<std::vector<uint>>& clusters) {
    for (const auto& c : clusters) {
      std::cout << "  {";
      for (auto it = c.begin(); it != c.end(); ++it) {
        if (it != c.begin()) std::cout << ", ";
        std::cout << *it;
      }
      std::cout << "}\n";
    }
  };

  // Window 0: expect cluster {1,2,3}
  {
    const auto& clusters = cpibt.get_clusters(0);  // t=0 → window 0
    bool found_123 = false;
    for (const auto& c : clusters)
      if (c == std::vector<uint>{1,2,3}) found_123 = true;
    assert(found_123 && "window 0: cluster {1,2,3} not found");
    std::cout << "window 0 clusters (" << clusters.size() << "):\n";
    print_clusters(clusters);
  }

  // Window 1: expect clusters {2,4} and {1,5}
  {
    const auto& clusters = cpibt.get_clusters(2);  // t=2 → window 1
    bool found_24 = false, found_15 = false;
    for (const auto& c : clusters) {
      if (c == std::vector<uint>{2,4}) found_24 = true;
      if (c == std::vector<uint>{1,5}) found_15 = true;
    }
    assert(found_24 && "window 1: cluster {2,4} not found");
    assert(found_15 && "window 1: cluster {1,5} not found");
    std::cout << "window 1 clusters (" << clusters.size() << "):\n";
    print_clusters(clusters);
  }

  std::cout << "test_clusters PASSED\n";
}

static void test_init_cluster_trees_order()
{
  std::mt19937 MT(0);
  const Instance ins("./assets/random-11-5.map", &MT, 4);
  DistTable D(&ins);

  // Build a root ClusteredHNode (constructor sets order via initialize_order;
  // we overwrite it afterwards).
  Config C = ins.starts;
  ClusteredHNode node(C, D, nullptr, 0, 0);

  // Overwrite order: agent 3 highest priority, then 1, then 2, then 0.
  node.order = {3, 1, 2, 0};

  // One cluster with all 4 agents so that pop_priority_lnode traverses all
  // depths (depth 1→4) and we see every agent's _who in the returned LNodes.
  std::vector<Cluster> clusters(1);
  clusters[0].agents = {0, 1, 2, 3};

  node.init_cluster_trees(clusters);

  // Verify the cluster order was built from node.order: expect [3,1,2,0].
  const auto& ct_order = node.cluster_trees[0].order;
  assert(ct_order == (std::vector<uint>{3, 1, 2, 0}) &&
         "cluster order should be (3,1,2,0)");

  // Pop all LNodes and collect _who for depth >= 1 nodes (depth-0 roots have
  // no meaningful _who).  Unique values in order of first appearance must
  // match node.order = {3,1,2,0}: the priority-based selection ensures the
  // highest-priority agent's subtree is always exhausted before the next.
  std::vector<uint> seen_order;
  std::unordered_set<uint> seen_set;
  while (true) {
    auto L = node.pop_priority_lnode();
    if (!L) break;
    if (L->depth >= 1) {
      if (seen_set.insert(L->_who).second)
        seen_order.push_back(L->_who);
    }
  }

  assert(seen_order == (std::vector<uint>{3, 1, 2, 0}) &&
         "pop order should match node.order: (3,1,2,0)");

  std::cout << "test_init_cluster_trees_order PASSED\n";
}




int main(int argc, char* argv[])
{
  // arguments parser
  argparse::ArgumentParser program("lacam2", "0.1.0");
  program.add_argument("-m", "--map").help("map file").required();
  program.add_argument("-i", "--scen")
      .help("scenario file")
      .default_value(std::string(""));
  program.add_argument("-N", "--num").help("number of agents").required();
  program.add_argument("-s", "--seed")
      .help("seed")
      .default_value(std::string("0"));
  program.add_argument("-v", "--verbose")
      .help("verbose")
      .default_value(std::string("0"));
  program.add_argument("-t", "--time_limit_sec")
      .help("time limit sec")
      .default_value(std::string("3"));
  program.add_argument("-o", "--output")
      .help("output file")
      .default_value(std::string("./build/result.txt"));
  program.add_argument("-l", "--log_short")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-O", "--objective")
      .help("0: none, 1: makespan, 2: sum_of_loss")
      .default_value(std::string("0"))
      .action([](const std::string& value) {
        static const std::vector<std::string> C = {"0", "1", "2"};
        if (std::find(C.begin(), C.end(), value) != C.end()) return value;
        return std::string("0");
      });
  program.add_argument("-r", "--restart_rate")
      .help("restart rate")
      .default_value(std::string("0.001"));
  program.add_argument("-od", "--odplanner")
      .help("use ODPlanner (Operator Decomposition A*)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-rand_planner", "--rand_planner")
      .help("use RandomPlanner (random PIBT rollouts from initial state)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-cplanner", "--cplanner")
      .help("use ClusteredPlanner (runs 5000 ClusterDetectionPIBT rollouts and prints clusters)")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-cem", "--cem")
      .help("use CEMPlanner (build policy from random rollouts, then run one PolicyPIBT rollout)")
      .default_value(false)
      .implicit_value(true);
    program.add_argument("-max_ll")
      .help("max allowed low-level search; -1 disables cutoff")
      .default_value(std::string("-1"));
      program.add_argument("-max_ll_decay")
        .help("decay factor for ancestor max_ll when a node exceeds its budget")
        .default_value(std::string("1.0"));
  program.add_argument("--test")
      .help("run unit tests and exit")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("--gui")
      .help("launch GUI visualizer (map + agents + goals)")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_known_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  if (program.get<bool>("test")) {
    test_clusters();
    test_init_cluster_trees_order();
    return 0;
  }

  // setup instance
  const auto verbose = std::stoi(program.get<std::string>("verbose"));
  const auto time_limit_sec =
      std::stoi(program.get<std::string>("time_limit_sec"));
  const auto scen_name = program.get<std::string>("scen");
  const auto seed = std::stoi(program.get<std::string>("seed"));
  auto MT = std::mt19937(seed);
  const auto map_name = program.get<std::string>("map");
  const auto output_name = program.get<std::string>("output");
  const auto log_short = program.get<bool>("log_short");
  const auto N1 = std::stoi(program.get<std::string>("num"));
  const auto ins1 = scen_name.size() > 0 ? Instance(scen_name, map_name, N1)
                                        : Instance(map_name, &MT, N1);

  const auto ins = ins1.multiply(1);
  const auto N = N1 * 1;



  const auto objective =
      static_cast<Objective>(std::stoi(program.get<std::string>("objective")));
  const auto restart_rate = std::stof(program.get<std::string>("restart_rate"));
  const auto use_odplanner = program.get<bool>("odplanner");
  const auto use_rand_planner = program.get<bool>("rand_planner");
  const auto use_cplanner = program.get<bool>("cplanner");
  const auto use_cem = program.get<bool>("cem");
  const auto max_ll_decay = std::clamp(std::stof(program.get<std::string>("max_ll_decay")), 0.0f, 1.0f);
  Planner::max_ll = std::stoi(program.get<std::string>("max_ll"));
  Planner::max_ll_decay = max_ll_decay;
  if (!ins.is_valid(1)) return 1;

  if (program.get<bool>("gui")) {
    run_gui(ins);
    return 0;
  }

  // solve
  auto additional_info = std::string("");
  const auto deadline = Deadline(time_limit_sec * 1000);
  Solution solution;
  if (use_cem) {
    CEMPlanner prs_planner(&ins, &deadline, &MT, verbose);
    solution = prs_planner.solve(additional_info);
  } else if (use_cplanner) {
    auto window = 4;
    auto rollouts = 1000;
    ClusteredPlanner cplanner(&ins, &deadline, &MT, verbose, rollouts, window);
    solution = cplanner.solve();
  } else if (use_rand_planner) {
    RandomPlanner rand_planner(&ins, &deadline, &MT, verbose);
    solution = rand_planner.solve();
  } else if (use_odplanner) {
    ODPlanner od_planner(&ins, &deadline, &MT, verbose - 1);
    solution = od_planner.solve(additional_info);
  } else {
    solution = solve(ins, additional_info, verbose - 1, &deadline, &MT,
                     objective, restart_rate);
  }
  const auto comp_time_ms = deadline.elapsed_ms();

  // failure
  if (solution.empty()) info(1, verbose, "failed to solve");

  // check feasibility
  if (!is_feasible_solution(ins, solution, verbose)) {
    info(0, verbose, "invalid solution");
  } else {
    info(0, verbose, "Solution is valid. Len: ", solution.size());

  }

  // post processing
  print_stats(verbose, ins, solution, comp_time_ms);
  make_log(ins, solution, output_name, comp_time_ms, map_name, seed,
           additional_info, log_short);
  return 0;
}
