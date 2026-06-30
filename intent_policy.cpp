#include "intent_policy.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float my_atan2(float y, float x) { return std::atan2(y, x); }
static float my_sqrt(float v)           { return std::sqrt(v); }

// ── helpers ───────────────────────────────────────────────────────────────────

// Assign angle sector 0-7 (0=E,1=NE,2=N,...,7=SE CCW from East, +y=up)
static inline int angle_sector(int dx, int dy)
{
    // negate dy: grid y increases downward, we want +y = North for the compass
    float a = my_atan2(-(float)dy, (float)dx);
    if (a < 0.f) a += 2.f * (float)M_PI;
    return (int)(a / ((float)M_PI / 4.f)) % 8;
}

// Manhattan-distance ring index, -1 if out of range
static inline int manhattan_ring(int dx, int dy)
{
    int md = std::abs(dx) + std::abs(dy);
    if (md < 1)  return -1;
    if (md <= 1)  return 0;
    if (md <= 4)  return 1;
    if (md <= 10) return 2;
    if (md <= 20) return 3;
    return -1;
}

// ── AgentFeatures ─────────────────────────────────────────────────────────────

std::array<float, FEATURE_SIZE> AgentFeatures::flat() const
{
    std::array<float, FEATURE_SIZE> out{};
    int k = 0;
    out[k++] = pos_radius;
    out[k++] = pos_angle_x;
    out[k++] = pos_angle_y;
    out[k++] = 0.f;
    out[k++] = intent_x;
    out[k++] = intent_y;
    for (int a = 0; a < N_ANGLES; ++a)
        for (int r = 0; r < N_RINGS; ++r)
            for (int c = 0; c < N_CHANNELS; ++c)
                out[k++] = radar[a][r][c];
    return out;
}

std::string AgentFeatures::summary(uint agent_id, int cx, int cy) const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "Agent #" << agent_id << " (" << cx << "," << cy << ")\n";
    ss << "--- Positional ---\n";
    ss << "radius  : " << pos_radius  << "\n";
    ss << "angle_x : " << pos_angle_x << "\n";
    ss << "angle_y : " << pos_angle_y << "\n";
    ss << "--- Intent self ---\n";
    ss << "intent_x: " << intent_x << "\n";
    ss << "intent_y: " << intent_y << "\n";
    ss << "--- Radar ---\n";
    ss << "(select a ring btn)\n";
    return ss.str();
}

std::string AgentFeatures::ring_detail(int ring) const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "Ring " << ring << " " << RING_LABELS[ring] << "\n";
    ss << "Dir   Flow  Thrt  Cross\n";
    for (int a = 0; a < N_ANGLES; ++a) {
        ss << std::left << std::setw(5) << ANGLE_NAMES[a]
           << " " << radar[a][ring][0]
           << " " << radar[a][ring][1]
           << " " << radar[a][ring][2] << "\n";
    }
    return ss.str();
}

// ── PolicyController ──────────────────────────────────────────────────────────

PolicyController::PolicyController(const Instance& ins)
    : ins_(ins), dist_(&ins), ctx_(ins.N)
{
    dist_at_start_.resize(ins.N);
    for (uint i = 0; i < ins.N; ++i)
        dist_at_start_[i] = dist_.get(i, const_cast<Vertex*>(ins.starts[i]));
    // NaN = no override
    intent_overrides_.assign(ins.N, {std::numeric_limits<float>::quiet_NaN(),
                                      std::numeric_limits<float>::quiet_NaN()});
}

void PolicyController::set_learning_mode(const Solution& solution)
{
    if (solution.empty()) {
        mode_     = PolicyMode::Execution;
        solution_ = nullptr;
    } else {
        mode_     = PolicyMode::Learning;
        solution_ = &solution;
    }
}

// BFS from src, max BFS depth = max_depth. Returns array indexed by vertex id.
std::vector<int> PolicyController::bfs_from(const Vertex* src, int max_depth) const
{
    std::vector<int> dist(ins_.G.size(), -1);
    std::queue<const Vertex*> q;
    dist[src->id] = 0;
    q.push(src);
    while (!q.empty()) {
        const Vertex* v = q.front(); q.pop();
        int d = dist[v->id];
        if (d >= max_depth) continue;
        for (Vertex* nb : v->neighbor) {
            if (dist[nb->id] == -1) {
                dist[nb->id] = d + 1;
                q.push(nb);
            }
        }
    }
    return dist;
}

// ── Phase 1a: positional ──────────────────────────────────────────────────────

void PolicyController::compute_positional_(uint i, const Config& cfg)
{
    const uint W = ins_.G.width;
    AgentFeatures& f = ctx_[i];

    const Vertex* curr = cfg[i];
    const Vertex* goal = ins_.goals[i];
    const int cx = static_cast<int>(curr->index % W);
    const int cy = static_cast<int>(curr->index / W);
    const int gx = static_cast<int>(goal->index % W);
    const int gy = static_cast<int>(goal->index / W);

    // pos_radius
    const uint d_start = dist_at_start_[i];
    if (d_start > 0) {
        uint d_curr = dist_.get(i, const_cast<Vertex*>(curr));
        f.pos_radius = static_cast<float>(d_curr) / static_cast<float>(d_start);
    }

    // pos_angle (coarse)
    float dx = static_cast<float>(gx - cx);
    float dy = static_cast<float>(gy - cy);
    float mag = std::sqrt(dx*dx + dy*dy);
    if (mag > 1e-4f) {
        auto coarse = [](float v){ return std::round(v * 10.f) / 10.f; };
        f.pos_angle_x = coarse(dx / mag);
        f.pos_angle_y = coarse(dy / mag);
    }
}

// ── Phase 1b: intent ─────────────────────────────────────────────────────────

void PolicyController::compute_intent_(uint i, const Config& cfg, int step)
{
    // If a manual override is set, always use it (skip plan-based computation)
    if (!std::isnan(intent_overrides_[i].ix)) {
        ctx_[i].intent_x = intent_overrides_[i].ix;
        ctx_[i].intent_y = intent_overrides_[i].iy;
        return;
    }

    if (mode_ != PolicyMode::Learning || solution_ == nullptr || step < 0)
        return;

    const uint W = ins_.G.width;
    const Solution& sol = *solution_;
    int future_t = std::min(step + INTENT_LENGTH, (int)sol.size() - 1);
    if (future_t <= step) return;

    const Vertex* curr = cfg[i];
    const Vertex* fut  = sol[static_cast<size_t>(future_t)][i];
    int cx = static_cast<int>(curr->index % W);
    int cy = static_cast<int>(curr->index / W);
    int fx = static_cast<int>(fut->index  % W);
    int fy = static_cast<int>(fut->index  / W);
    float dx = static_cast<float>(fx - cx);
    float dy = static_cast<float>(fy - cy);
    float mag = std::sqrt(dx*dx + dy*dy);
    if (mag > 1e-4f) {
        ctx_[i].intent_x = dx / mag;
        ctx_[i].intent_y = dy / mag;
    }
}

// ── Phase 2: radar ────────────────────────────────────────────────────────────

void PolicyController::compute_radar_(uint i, const Config& cfg,
                                      const std::vector<int>& bfs_dist)
{
    const uint W = ins_.G.width;
    AgentFeatures& f = ctx_[i];
    const Vertex* curr = cfg[i];
    const int cx = static_cast<int>(curr->index % W);
    const int cy = static_cast<int>(curr->index / W);

    // Count: total reachable cells and agent hits per (angle, ring, channel)
    int   total[N_ANGLES][N_RINGS]            = {};
    float accum[N_ANGLES][N_RINGS][N_CHANNELS] = {};

    // Walk every passable vertex reachable within BFS range
    for (const Vertex* v : ins_.G.V) {
        if (bfs_dist[v->id] == -1) continue;  // unreachable
        int vx = static_cast<int>(v->index % W);
        int vy = static_cast<int>(v->index / W);
        int dx = vx - cx, dy = vy - cy;
        int ring = manhattan_ring(dx, dy);
        if (ring < 0) continue;
        int sec = angle_sector(dx, dy);
        total[sec][ring]++;
    }

    // Walk other agents, accumulate intent alignment
    const float ix = f.intent_x, iy = f.intent_y;
    for (uint j = 0; j < (uint)cfg.size(); ++j) {
        if (j == i) continue;
        const Vertex* vj = cfg[j];
        if (bfs_dist[vj->id] == -1) continue;  // unreachable
        int vx = static_cast<int>(vj->index % W);
        int vy = static_cast<int>(vj->index / W);
        int dx = vx - cx, dy = vy - cy;
        int ring = manhattan_ring(dx, dy);
        if (ring < 0) continue;
        int sec = angle_sector(dx, dy);

        float jx = ctx_[j].intent_x, jy = ctx_[j].intent_y;
        float dot = ix * jx + iy * jy;
        if      (dot >  0.5f) accum[sec][ring][0] += 1.f;  // flow
        else if (dot < -0.5f) accum[sec][ring][1] += 1.f;  // threat
        else                  accum[sec][ring][2] += 1.f;  // cross
    }

    // Normalise by total valid cells per sector
    for (int a = 0; a < N_ANGLES; ++a)
        for (int r = 0; r < N_RINGS; ++r)
            if (total[a][r] > 0)
                for (int c = 0; c < N_CHANNELS; ++c)
                    f.radar[a][r][c] = accum[a][r][c] / (float)total[a][r];
}

// ── compute() ────────────────────────────────────────────────────────────────

void PolicyController::compute(const Config& cfg, int step)
{
    // Phase 1: positional + intent for ALL agents first
    for (uint i = 0; i < ins_.N; ++i) {
        ctx_[i] = AgentFeatures{};
        compute_positional_(i, cfg);
        compute_intent_(i, cfg, step);
    }
    // Phase 2: radar (needs all intents ready)
    for (uint i = 0; i < ins_.N; ++i) {
        auto bfs = bfs_from(cfg[i], MAX_RING_DIST);
        compute_radar_(i, cfg, bfs);
    }
}

// ── set_intent() / clear_intent_override() ───────────────────────────────────

void PolicyController::set_intent(uint agent_id, float ix, float iy)
{
    if (agent_id >= ctx_.size()) return;
    float mag = std::sqrt(ix * ix + iy * iy);
    if (mag > 1e-4f) {
        intent_overrides_[agent_id] = {ix / mag, iy / mag};
        ctx_[agent_id].intent_x = ix / mag;
        ctx_[agent_id].intent_y = iy / mag;
    } else {
        intent_overrides_[agent_id] = {0.f, 0.f};
        ctx_[agent_id].intent_x = 0.f;
        ctx_[agent_id].intent_y = 0.f;
    }
}

void PolicyController::clear_intent_override(uint agent_id)
{
    if (agent_id >= intent_overrides_.size()) return;
    intent_overrides_[agent_id] = {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::quiet_NaN()};
}

// ── compute_ring_map() ───────────────────────────────────────────────────────

RingMap PolicyController::compute_ring_map(uint agent_id,
                                           const Config& cfg) const
{
    RingMap result;
    const uint W = ins_.G.width;
    const Vertex* curr = cfg[agent_id];
    const int cx = static_cast<int>(curr->index % W);
    const int cy = static_cast<int>(curr->index / W);

    auto bfs = bfs_from(curr, MAX_RING_DIST);

    for (const Vertex* v : ins_.G.V) {
        if (bfs[v->id] == -1) continue;
        int vx = static_cast<int>(v->index % W);
        int vy = static_cast<int>(v->index / W);
        int dx = vx - cx, dy = vy - cy;
        int ring = manhattan_ring(dx, dy);
        if (ring < 0) continue;
        int sec = angle_sector(dx, dy);
        result[ring][sec].push_back(v->index);  // store grid index y*W+x
    }
    return result;
}
