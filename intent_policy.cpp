#include "intent_policy.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

// ── AgentFeatures::flat() ────────────────────────────────────────────────────

std::array<float, FEATURE_SIZE> AgentFeatures::flat() const
{
    std::array<float, FEATURE_SIZE> out{};
    int i = 0;
    out[i++] = pos_radius;
    out[i++] = pos_angle_x;
    out[i++] = pos_angle_y;
    out[i++] = 0.f;  // reserved
    out[i++] = intent_x;
    out[i++] = intent_y;
    for (int a = 0; a < N_ANGLES; ++a)
        for (int r = 0; r < N_RINGS; ++r)
            for (int c = 0; c < N_CHANNELS; ++c)
                out[i++] = radar[a][r][c];
    return out;
}

// ── AgentFeatures::summary() ─────────────────────────────────────────────────

static const char* ANGLE_NAMES[N_ANGLES] = {"E","NE","N","NW","W","SW","S","SE"};
static const char* RING_LABELS[N_RINGS]  = {"[1]","[2-4]","[5-10]","[11-20]"};

std::string AgentFeatures::summary(uint agent_id, int cx, int cy) const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "Agent #" << agent_id << "  pos(" << cx << "," << cy << ")\n";
    ss << "──── Positional ────\n";
    ss << "radius   : " << pos_radius  << "\n";
    ss << "angle_x  : " << pos_angle_x << "\n";
    ss << "angle_y  : " << pos_angle_y << "\n";
    ss << "──── Intent self ───\n";
    ss << "intent_x : " << intent_x << "\n";
    ss << "intent_y : " << intent_y << "\n";
    ss << "──── Radar ─────────\n";
    for (int a = 0; a < N_ANGLES; ++a) {
        for (int r = 0; r < N_RINGS; ++r) {
            ss << ANGLE_NAMES[a] << " " << RING_LABELS[r]
               << " F:" << radar[a][r][0]
               << " T:" << radar[a][r][1]
               << " X:" << radar[a][r][2] << "\n";
        }
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

void PolicyController::compute(const Config& cfg, int step)
{
    const uint W = ins_.G.width;

    for (uint i = 0; i < ins_.N; ++i) {
        AgentFeatures& f = ctx_[i];
        f = AgentFeatures{};  // zero all (intent stays 0 unless set below)

        const Vertex* curr = cfg[i];
        const Vertex* goal = ins_.goals[i];

        const int cx = static_cast<int>(curr->index % W);
        const int cy = static_cast<int>(curr->index / W);
        const int gx = static_cast<int>(goal->index % W);
        const int gy = static_cast<int>(goal->index / W);

        // ── pos_radius ────────────────────────────────────────────────────
        const uint d_start = dist_at_start_[i];
        if (d_start > 0) {
            uint d_curr = dist_.get(i, const_cast<Vertex*>(curr));
            f.pos_radius = static_cast<float>(d_curr) /
                           static_cast<float>(d_start);
        }

        // ── pos_angle (coarse euclidean ray to goal) ───────────────────────
        {
            float dx  = static_cast<float>(gx - cx);
            float dy  = static_cast<float>(gy - cy);
            float mag = std::sqrt(dx * dx + dy * dy);
            if (mag > 1e-4f) {
                auto coarse = [](float v) {
                    return std::round(v * 10.f) / 10.f;
                };
                f.pos_angle_x = coarse(dx / mag);
                f.pos_angle_y = coarse(dy / mag);
            }
        }

        // ── intent (learning mode only) ───────────────────────────────────
        if (mode_ == PolicyMode::Learning && solution_ != nullptr && step >= 0) {
            const Solution& sol = *solution_;
            int future_t = std::min(step + INTENT_LENGTH,
                                    static_cast<int>(sol.size()) - 1);
            // Only compute if agent actually moves between step and future_t
            if (future_t > step) {
                const Vertex* fut = sol[static_cast<size_t>(future_t)][i];
                int fx = static_cast<int>(fut->index % W);
                int fy = static_cast<int>(fut->index / W);
                float dx  = static_cast<float>(fx - cx);
                float dy  = static_cast<float>(fy - cy);
                float mag = std::sqrt(dx * dx + dy * dy);
                if (mag > 1e-4f) {
                    f.intent_x = dx / mag;
                    f.intent_y = dy / mag;
                }
            }
        }

        // radar remains zero until intent vectors are propagated
    }
}
