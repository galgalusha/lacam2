#pragma once
/*
 * intent_policy.hpp
 *
 * Defines PolicyFeatures (per-agent MLP input vector) and PolicyController
 * which populates the positional features from a Config + Instance.
 * Intent features are declared here but left at zero until a trained MLP
 * feeds them in a later phase.
 *
 * Feature layout (total: 4 + 2 + 96 = 102 floats per agent)
 * ──────────────────────────────────────────────────────────
 * Positional (4 floats):
 *   [0]  pos_radius     – normalised BFS distance D[i][curr] / D[i][start]
 *   [1]  pos_angle_x    – coarse euclidean ray to goal, normalised X
 *   [2]  pos_angle_y    – coarse euclidean ray to goal, normalised Y
 *   [3]  (reserved / padding to align intent block)
 *
 * Intent – self (2 floats):
 *   [4]  intent_x       – smoothed intent compass X  (zero until MLP ready)
 *   [5]  intent_y       – smoothed intent compass Y  (zero until MLP ready)
 *
 * Intent – radar (96 floats = 8 angles × 4 rings × 3 channels):
 *   Layout: radar[angle][ring][channel]
 *   angle  : 0=E 1=NE 2=N 3=NW 4=W 5=SW 6=S 7=SE  (CCW from East)
 *   ring   : 0=[1] 1=[2-4] 2=[5-10] 3=[11-20]   (Manhattan distance)
 *   channel: 0=flow  1=threat  2=cross
 *   All zero until intent values are available.
 */

#include <array>
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

// How many steps ahead to look when computing intent in learning mode
static constexpr int INTENT_LENGTH = 5;

// Manhattan distance bands for each ring index
static constexpr int RING_MIN[N_RINGS] = {1, 2,  5, 11};
static constexpr int RING_MAX[N_RINGS] = {1, 4, 10, 20};

// ── AgentFeatures ─────────────────────────────────────────────────────────────

struct AgentFeatures {
    // ── positional ──────────────────────────────────────────────────────────
    float pos_radius  = 0.f;   // D[i][curr] / D[i][start], 0=at goal, 1=at start
    float pos_angle_x = 0.f;   // normalised euclidean ray to goal, X component
    float pos_angle_y = 0.f;   // normalised euclidean ray to goal, Y component

    // ── intent – self ────────────────────────────────────────────────────────
    float intent_x = 0.f;
    float intent_y = 0.f;

    // ── intent – radar [angle][ring][channel] ────────────────────────────────
    float radar[N_ANGLES][N_RINGS][N_CHANNELS] = {};

    // Flat read-only view of all floats (for MLP input)
    std::array<float, FEATURE_SIZE> flat() const;

    // Human-readable summary
    std::string summary(uint agent_id, int cx, int cy) const;
};

// ── PolicyFeatures ─────────────────────────────────────────────────────────────

using PolicyFeatures = std::vector<AgentFeatures>;

// ── PolicyController ──────────────────────────────────────────────────────────

enum class PolicyMode { Execution, Learning };

class PolicyController {
public:
    explicit PolicyController(const Instance& ins);

    // Switch to learning mode: store the solution so compute() can derive
    // intents from look-ahead.  Resets to Execution mode if solution is empty.
    void set_learning_mode(const Solution& solution);

    // Recompute features for every agent given the current config.
    // step: current timestep index into the solution (-1 when no plan loaded).
    // In Learning mode uses step+INTENT_LENGTH look-ahead to set intents.
    void compute(const Config& cfg, int step = -1);

    const PolicyFeatures& context() const { return ctx_; }
    PolicyMode            mode()    const { return mode_; }

private:
    const Instance& ins_;
    DistTable       dist_;
    PolicyFeatures  ctx_;
    PolicyMode      mode_ = PolicyMode::Execution;
    const Solution* solution_ = nullptr;   // non-owning, valid during learning

    // BFS distance from agent i's start position (cached once)
    std::vector<uint> dist_at_start_;
};
