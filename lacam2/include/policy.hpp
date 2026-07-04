#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

#include <random>
#include <unordered_map>
#include <vector>

struct HNode;

// Abstract base class for neighbor selection policy used in PolicyPIBT.
// Given a config, an agent id, and all neighbor vertices (including self),
// returns a score in [-0.9, 0] for each neighbor. -0.9 = most preferred.
// PolicyPIBT adds these scores as tie-breakers to D values.
class Policy {
 public:
  virtual ~Policy() = default;
  virtual std::unordered_map<Vertex*, float> get_neighbor_scores(
      const Config& C, uint agent_id, const Vertices& neighbors) = 0;
  // Returns agent IDs ordered from smallest to largest priority value.
  // Default implementation returns H->order unchanged.
  virtual std::vector<uint> get_agents_order(HNode* H) const;
};

// Naive policy: all zeros (no preference, equivalent to vanilla PIBT random tie-breaking).
class RandomPolicy : public Policy {
 public:
  explicit RandomPolicy(std::mt19937* _MT) : MT(_MT) {}
  std::unordered_map<Vertex*, float> get_neighbor_scores(
      const Config& C, uint agent_id, const Vertices& neighbors) override;

 private:
  std::mt19937* MT;
};

struct NeighborScores {
  std::unordered_map<Vertex*, int> scores;  // neighbor vertex -> score
};

struct AgentScores {
  std::unordered_map<Vertex*, NeighborScores> vertex_scores;  // from-vertex -> neighbor scores

  // Pre-computed random scores for vertices absent from vertex_scores.
  // Maps from-vertex -> (neighbor-vertex -> random score in [-0.9, 0]).
  std::unordered_map<Vertex*, std::unordered_map<Vertex*, float>> blind_score_map;

  // Flat list of every execution order observed at a specific vertex (for CEM priority).
  std::unordered_map<Vertex*, std::vector<uint>> priority_records;

  void record_move(Vertex* from, Vertex* to);
  // Populate blind_score_map for every vertex in the instance.
  void finish_recording(const Instance* ins, std::mt19937* MT);
  // For each vertex in vertex_scores, replace its neighbor scores with a
  // random permutation of [1..num_neighbors], keeping keys intact.
  void randomize_scores(std::mt19937* rng);
};

// Policy backed by learned neighbor scores.
class ScorePolicy : public Policy {
 public:
  explicit ScorePolicy(std::vector<AgentScores> agent_policies,
                               std::mt19937* _MT)
      : policies(std::move(agent_policies)), MT(_MT) {}

  std::unordered_map<Vertex*, float> get_neighbor_scores(
      const Config& C, uint agent_id, const Vertices& neighbors) override;

  // Call once after all rollouts are recorded to pre-compute blind scores.
  void finish_recording(const Instance* ins);

  // Replace one agent's blind_score_map with a fresh random draw.
  void randomize_agent_blind_scores(uint agent_idx, const Instance* ins,
                                    std::mt19937* rng);

  // Shuffle vertex_scores for one agent with random [1..n] permutations.
  void randomize_agent_scores(uint agent_idx, std::mt19937* rng);

  // For every agent and every vertex in their vertex_scores, shuffle the
  // neighbor scores with a random permutation of [1..num_neighbors].
  void randomize_all_scores(std::mt19937* rng);

  uint random_count = 0;
  uint deterministic_count = 0;

  const std::vector<AgentScores>& get_policies() const { return policies; }

 private:
  std::vector<AgentScores> policies;
  std::mt19937* MT;
};

// ---------------------------------------------------------------------------
// CEM (Cross-Entropy Method) types
// ---------------------------------------------------------------------------

// Gaussian parameters for continuous execution-priority distribution.
struct PriorityDist {
  double mu;     // mean execution order
  double sigma;  // exploration noise (standard deviation)
};

// Per-agent probability distribution over neighbor choices.
// vertex_probs[v][u] = probability of choosing u when at vertex v.
// Probabilities for each vertex v sum to 1.0.
struct AgentProbabilityPolicy {
  std::unordered_map<Vertex*, std::unordered_map<Vertex*, double>> vertex_probs;
  std::unordered_map<Vertex*, PriorityDist> priority_dist;
};

// Convert an AgentPolicy (integer move counts) to normalized probabilities.
// Only vertices present in ap.vertex_scores are stored; blind-spot vertices
// are handled lazily by CrossEntropyPolicy (returns 0.0 == no preference).
AgentProbabilityPolicy to_probability_policy(const AgentScores& ap);

// Policy that samples a neighbor from per-agent probability distributions on every call.
// The sampled neighbor gets score -0.9; all others get 0.
// Also serves as the plain container for per-agent probability data.
// Used as the master policy in CEM; passed to PolicyRandomizer to sample
// DeterministicPolicy instances for rollout evaluation.
class ProbabilityPolicy : public Policy {
 public:
  // Default constructor (empty container; get_neighbor_scores must not be called).
  ProbabilityPolicy() : probs(), ins(nullptr), rng(nullptr) {}

  // Container-only constructor (ins/rng left null; get_neighbor_scores must not be called).
  explicit ProbabilityPolicy(size_t n) : probs(n), ins(nullptr), rng(nullptr) {}

  // Full constructor for use as a Policy.
  ProbabilityPolicy(std::vector<AgentProbabilityPolicy> prob_policies,
                    const Instance* ins,
                    std::mt19937* rng)
      : probs(std::move(prob_policies)), ins(ins), rng(rng) {}

  std::unordered_map<Vertex*, float> get_neighbor_scores(
      const Config& C, uint agent_id, const Vertices& neighbors) override;

  AgentProbabilityPolicy& operator[](size_t i) { return probs[i]; }
  const AgentProbabilityPolicy& operator[](size_t i) const { return probs[i]; }
  size_t size() const { return probs.size(); }

  std::vector<AgentProbabilityPolicy> probs;

 private:
  const Instance* ins;
  std::mt19937* rng;
};

// Per-agent discrete policy: for each vertex, a pre-calculated score for every
// valid neighbor. Scores are in [-0.9, 0], lower = more preferred.
struct AgentDeterministicPolicy {
  std::unordered_map<Vertex*, std::unordered_map<Vertex*, float>> rankings;  // vertex -> (neighbor -> score)
  std::vector<double> priority_grid;  // indexed by vertex->index (width*y+x), size = width*height
};


// On a known vertex: assigns the stored scores to neighbors.
// On a blind-spot vertex: returns 0.0 for all neighbors (random tie-breaking by PolicyPIBT).
// `discrete` is public so callers can move it out after rollout.
class DeterministicPolicy : public Policy {
 public:
  // Override: orders agents by priority_grid value at their current vertex (ascending).
  std::vector<uint> get_agents_order(HNode* H) const override;

  DeterministicPolicy(std::vector<AgentDeterministicPolicy> discrete_policies,
                      const Instance* ins,
                      std::mt19937* rng)
      : discrete(std::move(discrete_policies)),
        ins(ins),
        rng(rng) {}

  std::unordered_map<Vertex*, float> get_neighbor_scores(
      const Config& C, uint agent_id, const Vertices& neighbors) override;

  std::vector<AgentDeterministicPolicy> discrete;

 private:
  const Instance* ins;
  std::mt19937* rng;
};

// Samples num_policies complete DeterministicPolicy instances from a full ProbabilityPolicy.
// rngs[i % rngs.size()] is used both for sampling probabilities and as the blind-spot RNG
// of the resulting DeterministicPolicy, so each policy is independently seeded.
class PolicyRandomizer {
 public:  PolicyRandomizer(const Graph* graph = nullptr,
                   std::mt19937* rng = nullptr,
                   double priority_temprature = 1.0)
      : graph(graph), rng(rng), priority_temprature(priority_temprature) {}
  std::vector<std::shared_ptr<DeterministicPolicy>> operator()(
      const ProbabilityPolicy& prob_policy,
      const Instance* ins,
      std::vector<std::mt19937>& rngs,
      uint num_policies) const;

  AgentDeterministicPolicy sample_agent_policy(
      const AgentProbabilityPolicy& prob_policy,
      const std::vector<PriorityDist>& filled_dist) const;

  const Graph* graph = nullptr;
  std::mt19937* rng = nullptr;
  double priority_temprature;
};

void test_randomizer();
