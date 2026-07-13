#pragma once

#include "instance.hpp"
#include "policy.hpp"

#include <chrono>

// ── Global UI state ──────────────────────────────────────────────────────────
extern uint   g_status_lines;       // how many lines the last status block occupied
extern int    g_probe_agent;        // agent whose entropy stats are shown
extern double g_last_randomizer_ms; // time spent in randomizer last gen
extern double g_last_rollouts_ms;   // time spent in rollout loop last gen

// ── Outer-loop context (passed by solve() to solve_with_cem()) ───────────────
// When gen == UINT_MAX the outer panel is hidden (standalone mode).
struct OuterContext {
  uint gen                                          = UINT_MAX;
  std::chrono::steady_clock::time_point start_time = {};
  uint complementary_soc                            = 0;
};

// ── Key events ───────────────────────────────────────────────────────────────
enum class KeyEvent {
  None,
  Char,
  StopInner,   // Space  — stops the current solve_with_cem loop
  StopOuter,   // ESC   — stops the outer solve loop
  ShiftUp, ShiftDown, ShiftLeft, ShiftRight,
};

struct KeyResult {
  KeyEvent event = KeyEvent::None;
  char     ch    = 0;
};

// Non-blocking: returns immediately. Detects regular keys and Shift+Arrow sequences.
KeyResult read_key_event_nonblocking();

// Blocking read of a single raw key (for two-key sequences like 'l'+'p').
char read_key_blocking();

// Update map viewport offset based on a Shift+Arrow scroll event.
// Also forces the map to redraw at the next draw_cem_status call.
void cem_ui_handle_scroll(KeyEvent event);

// Redraws the CEM status box and (every MAP_RENDER_INTERVAL s) the agent map.
// If outer.gen != UINT_MAX an additional outer-loop panel is drawn on top.
void draw_cem_status(uint gen, double elapsed_sec,
                     int elite_size, int new_global_elite, uint best_cost,
                     const ProbabilityPolicy& prob_policy, int num_agents,
                     const Instance* ins,
                     const OuterContext& outer = OuterContext{});

// Interactive prompts (blocking; restores the terminal after returning).
void prompt_agent_id(int num_agents);
void prompt_gen_laplace();
bool prompt_base_lr();   // returns true if a reset was requested
