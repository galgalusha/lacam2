#include "../include/cem_ui.hpp"
#include "../include/cem_params.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <termios.h>
#include <unistd.h>

// ── Global UI state ───────────────────────────────────────────────────────────────
uint   g_status_lines       = 0;
int    g_probe_agent        = 0;
double g_last_randomizer_ms = 0.0;
double g_last_rollouts_ms   = 0.0;

// ── Map viewport constants & state ────────────────────────────────────────────────────
static constexpr double MAP_RENDER_INTERVAL = 2.0;  // seconds between map redraws
static constexpr int    MAP_VIEW_W          = 80;   // grid columns visible
static constexpr int    MAP_VIEW_H          = 22;   // grid rows visible
static constexpr int    MAP_SCROLL_STEP     = 5;    // cells per Shift+Arrow press

static int  s_map_off_x      = 0;
static int  s_map_off_y      = 0;
static auto s_last_map_time  = std::chrono::steady_clock::time_point{};

static uint s_status_box_lines = 0;  // lines printed by the status box alone
static uint s_map_lines        = 0;  // lines printed by the map alone

// ── Terminal RAII guard ──────────────────────────────────────────────────────────────
struct TermGuard {
  struct termios saved_term;
  int            saved_flags;

  TermGuard(bool raw, bool blocking) {
    tcgetattr(STDIN_FILENO, &saved_term);
    saved_flags = fcntl(STDIN_FILENO, F_GETFL, 0);

    struct termios t = saved_term;
    if (raw) t.c_lflag &= ~(ICANON | ECHO);
    else     t.c_lflag |=  (ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &t);

    int fl = saved_flags;
    if (blocking) fl &= ~O_NONBLOCK;
    else          fl |=  O_NONBLOCK;
    fcntl(STDIN_FILENO, F_SETFL, fl);
  }

  ~TermGuard() {
    fcntl(STDIN_FILENO, F_SETFL, saved_flags);
    tcsetattr(STDIN_FILENO, TCSANOW, &saved_term);
  }
};

// ── Key input ────────────────────────────────────────────────────────────────────────────

KeyResult read_key_event_nonblocking()
{
  TermGuard guard(/*raw=*/true, /*blocking=*/false);

  KeyResult res;
  char c = 0;
  if (::read(STDIN_FILENO, &c, 1) != 1)
    return res;  // KeyEvent::None

  if (c == ' ') {
    res.event = KeyEvent::Stop;
    res.ch    = ' ';
    return res;
  }

  if (c == 0x1B) {
    // Drain the rest of the escape sequence (bytes from one key press arrive
    // together in the TTY buffer, so non-blocking reads succeed immediately).
    char buf[8] = {};
    int  n      = 0;
    while (n < static_cast<int>(sizeof(buf)) - 1) {
      char x = 0;
      if (::read(STDIN_FILENO, &x, 1) != 1) break;
      buf[n++] = x;
    }
    if (n == 0) {
      // Bare ESC → stop
      res.event = KeyEvent::Stop;
      res.ch    = 0x1B;
      return res;
    }
    // Shift+Arrow:  ESC [ 1 ; 2 X   (buf = "[1;2X", n≥5)
    if (n >= 5 && buf[0] == '[' && buf[1] == '1'
              && buf[2] == ';' && buf[3] == '2') {
      switch (buf[4]) {
        case 'A': res.event = KeyEvent::ShiftUp;    return res;
        case 'B': res.event = KeyEvent::ShiftDown;  return res;
        case 'C': res.event = KeyEvent::ShiftRight; return res;
        case 'D': res.event = KeyEvent::ShiftLeft;  return res;
      }
    }
    // Unknown sequence → ignore
    return res;  // KeyEvent::None
  }

  res.event = KeyEvent::Char;
  res.ch    = c;
  return res;
}

char read_key_blocking()
{
  TermGuard guard(/*raw=*/true, /*blocking=*/true);
  char c = 0;
  ::read(STDIN_FILENO, &c, 1);
  return c;
}

// ── Map scroll ───────────────────────────────────────────────────────────────────────────
void cem_ui_handle_scroll(KeyEvent event)
{
  switch (event) {
    case KeyEvent::ShiftUp:    s_map_off_y -= MAP_SCROLL_STEP; break;
    case KeyEvent::ShiftDown:  s_map_off_y += MAP_SCROLL_STEP; break;
    case KeyEvent::ShiftLeft:  s_map_off_x -= MAP_SCROLL_STEP; break;
    case KeyEvent::ShiftRight: s_map_off_x += MAP_SCROLL_STEP; break;
    default: return;  // not a scroll event – don’t reset the timer
  }
  if (s_map_off_x < 0) s_map_off_x = 0;
  if (s_map_off_y < 0) s_map_off_y = 0;
  // Force the map to redraw on the next draw_cem_status call.
  s_last_map_time = std::chrono::steady_clock::time_point{};
}

// ── Map rendering helpers ──────────────────────────────────────────────────────────────

// UTF-8 arrow for direction (dx, dy) from a vertex toward its best neighbor.
static const char* direction_arrow(int dx, int dy)
{
  if (dx ==  1 && dy ==  0) return "\u2192";  // →
  if (dx == -1 && dy ==  0) return "\u2190";  // ←
  if (dx ==  0 && dy == -1) return "\u2191";  // ↑  (y=0 is top row)
  if (dx ==  0 && dy ==  1) return "\u2193";  // ↓
  return "o";                                  // stay-in-place / diagonal
}

// ANSI foreground color for a certainty probability.
static const char* prob_color(double p)
{
  if (p <  0.75) return "\033[2;31m";  // dim red (dark red)
  if (p <  0.90) return "\033[91m";    // bright red
  if (p <  0.95) return "\033[33m";    // yellow-orange
  if (p <  0.99) return "\033[93m";    // bright yellow
  return "\033[92m";                   // bright green (> 0.99)
}

// Render the agent map if MAP_RENDER_INTERVAL has elapsed since the last render.
// Prints directly to stdout and returns the number of lines emitted (0 if skipped).
static uint draw_agent_map(const Instance* ins, const ProbabilityPolicy& prob_policy)
{
  if (ins == nullptr) return 0;

  const auto now = std::chrono::steady_clock::now();
  const double dt = std::chrono::duration<double>(now - s_last_map_time).count();
  if (dt < MAP_RENDER_INTERVAL) return 0;
  s_last_map_time = now;

  // Erase the old map before drawing the new one.
  std::cout << "\033[J";

  const int gw = static_cast<int>(ins->G.width);
  const int gh = static_cast<int>(ins->G.height);

  // Clamp offsets to valid range.
  s_map_off_x = std::max(0, std::min(s_map_off_x, std::max(0, gw - 1)));
  s_map_off_y = std::max(0, std::min(s_map_off_y, std::max(0, gh - 1)));

  const int view_w = std::min(MAP_VIEW_W, gw - s_map_off_x);
  const int view_h = std::min(MAP_VIEW_H, gh - s_map_off_y);

  // Agent start / goal for g_probe_agent.
  Vertex* agent_start = (g_probe_agent < static_cast<int>(ins->N))
                        ? ins->starts[g_probe_agent] : nullptr;
  Vertex* agent_goal  = (g_probe_agent < static_cast<int>(ins->N))
                        ? ins->goals[g_probe_agent]  : nullptr;

  const AgentProbabilityPolicy* apol = nullptr;
  if (g_probe_agent < static_cast<int>(prob_policy.size()))
    apol = &prob_policy[g_probe_agent];

  uint lines = 0;
  auto pr = [&](const std::string& s) { std::cout << s << "\n"; ++lines; };

  // ── Header ─────────────────────────────────────────────────────────────────
  {
    std::ostringstream h;
    h << "  \033[2m\u2500\u2500 Map [agent=" << g_probe_agent
      << "]  view=" << view_w << "\u00d7" << view_h
      << "  col=" << s_map_off_x << "  row=" << s_map_off_y
      << "  [Shift+\u2190\u2192\u2191\u2193 scroll]\033[0m";
    pr(h.str());
  }

  // ── Grid rows ─────────────────────────────────────────────────────────────────
  for (int row = 0; row < view_h; ++row) {
    const int y = s_map_off_y + row;
    std::string line;
    line.reserve(view_w * 16);

    for (int col = 0; col < view_w; ++col) {
      const int  x   = s_map_off_x + col;
      const uint idx = static_cast<uint>(y * gw + x);
      Vertex* v = (idx < ins->G.U.size()) ? ins->G.U[idx] : nullptr;

      if (v == nullptr) {
        line += "\033[2m#\033[0m";    // dim '#' for obstacle
        continue;
      }
      if (v == agent_start) {
        line += "\033[1;32mS\033[0m"; // bold green S
        continue;
      }
      if (v == agent_goal) {
        line += "\033[1;36mG\033[0m"; // bold cyan G
        continue;
      }

      // Vertex with probability data: pick best neighbor, derive direction.
      if (apol != nullptr) {
        auto it = apol->vertex_probs.find(v);
        if (it != apol->vertex_probs.end() && !it->second.empty()) {
          Vertex* best_nb = nullptr;
          double  best_p  = -1.0;
          for (const auto& [nb, p] : it->second)
            if (p > best_p) { best_p = p; best_nb = nb; }

          if (best_nb != nullptr) {
            const int dx = static_cast<int>(best_nb->index % static_cast<uint>(gw))
                         - static_cast<int>(v->index    % static_cast<uint>(gw));
            const int dy = static_cast<int>(best_nb->index / static_cast<uint>(gw))
                         - static_cast<int>(v->index    / static_cast<uint>(gw));
            line += prob_color(best_p);
            line += direction_arrow(dx, dy);
            line += "\033[0m";
            continue;
          }
        }
      }

      // Passable vertex with no probability data.
      line += "\033[90m.\033[0m";
    }

    pr(line);
  }

  // ── Legend ──────────────────────────────────────────────────────────────────────
  {
    struct Bucket { const char* color; const char* label; };
    static const Bucket BUCKETS[] = {
      {"\033[2;31m", "<0.75"},
      {"\033[91m",   "0.75-0.90"},
      {"\033[33m",   "0.90-0.95"},
      {"\033[93m",   "0.95-0.99"},
      {"\033[92m",   ">0.99"},
    };
    std::string leg = "  ";
    for (const auto& b : BUCKETS) {
      leg += b.color;
      leg += "\u25a0\033[0m ";   // ■ + reset + space
      leg += "\033[2m";
      leg += b.label;
      leg += "\033[0m  ";
    }
    pr(leg);
  }

  std::cout << std::flush;
  return lines;
}

// ── Status display ───────────────────────────────────────────────────────────────────────

void draw_cem_status(uint gen, double elapsed_sec,
                     int elite_size, int new_global_elite, uint best_cost,
                     const ProbabilityPolicy& prob_policy, int num_agents,
                     const Instance* ins)
{
  // Move cursor to the top of the entire output region (status box + map).
  const uint total_prev = s_status_box_lines + s_map_lines;
  if (total_prev > 0)
    std::cout << "\033[" << total_prev << "A";

  const int mins = static_cast<int>(elapsed_sec / 60.0);
  const double secs = elapsed_sec - mins * 60.0;

  // ── Box geometry (100 cols wide) ──────────────────────────────────────────────────
  constexpr int LW = 24;
  constexpr int MW = 36;
  constexpr int RW = 30;
  constexpr int FW = LW + MW + RW + 6;

  auto rep = [](const char* g, int n) -> std::string {
    std::string s; for (int i = 0; i < n; ++i) s += g; return s;
  };
  auto pad = [](std::string s, int w) -> std::string {
    if ((int)s.size() > w) s.resize(w);
    else s.append(w - (int)s.size(), ' ');
    return s;
  };

  const std::string full_top  = "\u2554" + rep("\u2550", FW+2) + "\u2557";
  const std::string split_sep = "\u2560" + rep("\u2550", LW+2)
                              + "\u2566" + rep("\u2550", MW+2)
                              + "\u2566" + rep("\u2550", RW+2) + "\u2563";
  const std::string mid_sep   = "\u2560" + rep("\u2550", LW+2)
                              + "\u256c" + rep("\u2550", MW+2)
                              + "\u256c" + rep("\u2550", RW+2) + "\u2563";
  const std::string bot       = "\u255a" + rep("\u2550", LW+2)
                              + "\u2569" + rep("\u2550", MW+2)
                              + "\u2569" + rep("\u2550", RW+2) + "\u255d";

  auto row = [&](const std::string& L, const std::string& M, const std::string& R) -> std::string {
    return "\u2551 " + pad(L, LW) + " \u2551 " + pad(M, MW) + " \u2551 " + pad(R, RW) + " \u2551";
  };

  // ── Entropy data ─────────────────────────────────────────────────────────────────
  static constexpr double THRESH[4] = {0.80, 0.90, 0.95, 0.99};
  double cpct[4] = {};
  uint visited = 0;
  if (g_probe_agent < num_agents) {
    const auto& apol = prob_policy[g_probe_agent];
    visited = static_cast<uint>(apol.vertex_probs.size());
    if (visited > 0) {
      for (int ti = 0; ti < 4; ++ti) {
        uint cnt = 0;
        for (const auto& [v, nb] : apol.vertex_probs) {
          double best = 0.0;
          for (const auto& [u, p] : nb) if (p > best) best = p;
          if (best > THRESH[ti]) ++cnt;
        }
        cpct[ti] = 100.0 * cnt / visited;
      }
    }
  }

  // ── Format helpers ─────────────────────────────────────────────────────────────────
  auto fi = [](long long v, int w) -> std::string {
    std::ostringstream s; s << std::setw(w) << v; return s.str();
  };
  auto fd = [](double v, int w, int p) -> std::string {
    std::ostringstream s;
    s << std::fixed << std::setprecision(p) << std::setw(w) << v;
    return s.str();
  };

  auto erow = [&](int ti) -> std::string {
    std::ostringstream s;
    s << "  > " << std::fixed << std::setprecision(2) << THRESH[ti]
      << "  |  " << std::setw(6) << std::fixed << std::setprecision(2) << cpct[ti] << "%";
    return s.str();
  };

  // ── Header ──────────────────────────────────────────────────────────────────────────
  std::string gen_str = fi((long long)gen, 6);
  std::string soc_str = (best_cost == UINT_MAX) ? "      ---" : fi((long long)best_cost, 9);
  std::string time_s;
  {
    std::ostringstream t;
    t << mins << "m" << std::fixed << std::setprecision(1) << secs << "s";
    time_s = t.str();
  }
  const std::string hdr_vis = "GEN: " + gen_str + "   TIME: " + time_s
                             + "   BEST SoC: " + soc_str;
  int vis_len  = static_cast<int>(hdr_vis.size());
  int tot_pad  = FW - vis_len;
  int lpad_n   = (tot_pad > 0) ? tot_pad / 2 : 0;
  int rpad_n   = (tot_pad > 0) ? tot_pad - lpad_n : 0;
  const std::string hdr_bold =
      std::string(lpad_n, ' ')
    + "GEN: \033[1m" + gen_str + "\033[0m"
    + "   TIME: " + time_s
    + "   BEST SoC: \033[1m" + soc_str + "\033[0m"
    + std::string(rpad_n, ' ');

  // ── Print status box ──────────────────────────────────────────────────────────────
  // Each line is erased before printing so stale content from longer previous
  // lines doesn't bleed through (\033[2K = erase entire current line).
  uint lines = 0;
  auto pr = [&](const std::string& s) { std::cout << "\033[2K" << s << "\n"; ++lines; };

  pr(full_top);
  std::cout << "\033[2K\u2551 " << hdr_bold << " \u2551\n"; ++lines;
  pr(split_sep);
  pr(row("  ENTROPY  [agent =" + fi(g_probe_agent, 3) + " ]",
         "  GEN STATS",
         "  TIME STATS"));
  pr(row("  visited =" + fi((long long)visited, 8),
         "  elite_size  :" + fi(elite_size, 9),
         "  rng_ms      :" + fd(g_last_randomizer_ms, 13, 1)));
  pr(row("",
         "  new_elite   :" + fi(new_global_elite, 9),
         "  rollout_ms  :" + fd(g_last_rollouts_ms,   13, 1)));
  pr(mid_sep);
  pr(row("  thresh   |  confident",
         "  HYPER PARAMS",
         ""));
  pr(row("  " + std::string(20, '-'),
         "  lr          :" + fd(LEARNING_RATE,         13, 4),
         ""));
  pr(row(erow(0), "  lr_base     :" + fd(BASE_LEARNING_RATE,    13, 4), ""));
  pr(row(erow(1), "  gen_smooth  :" + fd(GEN_LAPLACE_SMOOTHING, 13, 4), ""));
  pr(row(erow(2), "", ""));
  pr(row(erow(3), "", ""));
  pr(bot);
  pr("  \033[2m[Spc/Esc]\033[0m stop  \033[2m[a]\033[0m agent  "
     "\033[2m[l+p]\033[0m smooth  \033[2m[l+r]\033[0m lr_base+reset  "
     "\033[2m[Shift+\u2190\u2192\u2191\u2193]\033[0m scroll map");

  std::cout << std::flush;
  s_status_box_lines = lines;

  // ── Agent map (throttled to MAP_RENDER_INTERVAL) ──────────────────────────────────
  // draw_agent_map returns 0 (skip) or N>0 (drew N lines after emitting \033[J).
  const uint new_map_lines = draw_agent_map(ins, prob_policy);
  if (new_map_lines > 0) {
    s_map_lines = new_map_lines;
  } else if (s_map_lines > 0) {
    // Map unchanged: skip cursor past it so the next draw starts from below.
    std::cout << "\033[" << s_map_lines << "B";
  }
  std::cout << std::flush;

  g_status_lines = s_status_box_lines + s_map_lines;
}

// ── Interactive prompts ──────────────────────────────────────────────────────────────────
void prompt_agent_id(int num_agents)
{
  TermGuard guard(/*raw=*/false, /*blocking=*/true);
  std::cout << "Enter agent id [0-" << (num_agents - 1) << "]: " << std::flush;
  std::string line;
  if (std::getline(std::cin, line)) {
    try {
      int id = std::stoi(line);
      if (id >= 0 && id < num_agents) {
        g_probe_agent = id;
        g_status_lines += 2;
      }
    } catch (...) {}
  }
}

void prompt_gen_laplace()
{
  TermGuard guard(/*raw=*/false, /*blocking=*/true);
  std::cout << "Enter new gen_laplace_smoothing (current=" << GEN_LAPLACE_SMOOTHING << "): " << std::flush;
  std::string line;
  if (std::getline(std::cin, line)) {
    try {
      double v = std::stod(line);
      if (v > 0.0) {
        GEN_LAPLACE_SMOOTHING = v;
        g_status_lines += 2;
      }
    } catch (...) {}
  }
}

bool prompt_base_lr()
{
  TermGuard guard(/*raw=*/false, /*blocking=*/true);
  std::cout << "Enter new base_learning_rate (current=" << BASE_LEARNING_RATE << "): " << std::flush;
  std::string line;
  bool reset = false;
  if (std::getline(std::cin, line)) {
    try {
      double v = std::stod(line);
      if (v > 0.0 && v <= 1.0) {
        BASE_LEARNING_RATE = v;
        g_status_lines += 2;
        reset = true;
      }
    } catch (...) {}
  }
  return reset;
}
