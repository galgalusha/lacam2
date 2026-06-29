#pragma once
/*
 * intent_policy.hpp
 *
 * Feature layout (total: 4 + 2 + 96 = 102 floats per agent)
 * ──────────────────────────────────────────────────────────
 * Positional (4 floats):
 *   [0]  pos_radius     – D[i][curr] / D[i][start]
 *   [1]  pos_angle_x    – coarse euclidean ray to goal, X
 *   [2]  pos_angle_y    – coarse euclidean ray to goal, Y
 *   [3]  (reserved)
 * Intent – self (2 floats):
 *   [4]  intent_x / [5] intent_y  – smoothed compass (zero until MLP ready)
 * Intent – radar (96 floats = 8 angles × 4 rings × 3 channels):
 *   radar[angle][ring][channel]
 *   angle  : 0=E 1=NE 2=N 3=NW 4=W 5=SW 6=S 7=SE  (CCW from East, +y=up)
 *   ring   : 0=[1] 1=[2-4] 2=[5-10] 3=[11-20]  (Manhattan distance)
 *   channel: 0=flow  1=threat  2=cross
 */

#include <array>
#include <queue>
#include <string>
#include <vector>

#include <dist_table.hpp>
#include <instance.hpp>

// ── constants ────────────────────────────────────────────────────────────────

static constexpr int N_ANGLES     = 8;
static constexpr int N_RINGS      = 4;
static constexpr int N_CHANNELS   = 3;
static constexpr int RADAR_SIZE   = N_ANGLES * N_RINGS * N_CHANNELS;  // 96
static constexpr int FEATURE_SIZE = 4 + 2 + RADAR_SIZE;               // 102
static constexpr int INTENT_LENGTH = 5;
static constexpr int MAX_RING_DIST = 20;  // max Manhattan distance in rings

// Manhattan distance bands for each ring index
static constexpr int RING_MIN[N_RINGS] = { 1, 2,  5, 11};
static constexpr int RING_MAX[N_RINGS] = { 1, 4, 10, 20};

static const char* ANGLE_NAMES[N_ANGLES] = {"E","NE","N","NW","W","SW","S","SE"};
static const char* RING_LABELS[N_RINGS]  = {"[1]","[2-4]","[5-10]","[11-20]"};

// ── RingMap ───────────────────────────────────────────────────────────────────
// [ring][sector] → list of vertex grid-indices (y*W+x) that are BFS-reachable.
// Used by the GUI to visualize radar rings on the map.
using RingMap = std::array<std::array<std::vector<uint>, N_ANGLES>, N_RINGS>;

// ── AgentFeatures ─────────────────────────────────────────────────────────────

struct AgentFeatures {
    float pos_radius  = 0.f;
    float pos_angle_x = 0.f;
    float pos_angle_y = 0.f;

    float intent_x = 0.f;
    float intent_y = 0.f;

    // radar[angle][ring][channel]
    float radar[N_ANGLES][N_RINGS][N_CHANNELS] = {};

    // Flat float array for MLP input
    std::array<float, FEATURE_SIZE> flat() const;

    // Summary without radar (radar shown via GUI ring buttons)
    std::string summary(uint agent_id, int cx, int cy) const;

    // Per-ring detail text (all 8 sectors for one ring)
    std::string ring_detail(int ring) const;
};

// ── PolicyFeatures ────────────────────────────────────────────────────────────

using PolicyFeatures = std::vector<AgentFeatures>;

// ── PolicyController ──────────────────────────────────────────────────────────

enum class PolicyMode { Execution, Learning };

class PolicyController {
public:
    explicit PolicyController(const Instance& ins);

    void set_learning_mode(const Solution& solution);

    // Recompute all features for every agent.
    // step: current timestep (-1 = no plan).
    void compute(const Config& cfg, int step = -1);

    // Compute RingMap for a single agent (GUI visualisation, called on demand).
    RingMap compute_ring_map(uint agent_id, const Config& cfg) const;

    const PolicyFeatures& context() const { return ctx_; }
    PolicyMode            mode()    const { return mode_; }

private:
    const Instance& ins_;
    DistTable       dist_;
    PolicyFeatures  ctx_;
    PolicyMode      mode_     = PolicyMode::Execution;
    const Solution* solution_ = nullptr;

    std::vector<uint> dist_at_start_;

    // BFS from src; returns per-vertex-id distances (-1 = unreachable/too far)
    std::vector<int> bfs_from(const Vertex* src, int max_depth) const;

    // Feature computation phases (called in order by compute())
    void compute_positional_(uint i, const Config& cfg);
    void compute_intent_    (uint i, const Config& cfg, int step);
    void compute_radar_     (uint i, const Config& cfg,
                             const std::vector<int>& bfs_dist);
};
