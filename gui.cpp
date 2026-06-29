#include "gui.hpp"
#include "intent_policy.hpp"

#include <SFML/Graphics.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------
static const std::array<sf::Color, 20> PALETTE = {
    sf::Color(230, 25,  75),
    sf::Color(60,  180, 75),
    sf::Color(255, 225, 25),
    sf::Color(0,   130, 200),
    sf::Color(245, 130, 48),
    sf::Color(145, 30,  180),
    sf::Color(70,  240, 240),
    sf::Color(240, 50,  230),
    sf::Color(210, 245, 60),
    sf::Color(250, 190, 212),
    sf::Color(0,   128, 128),
    sf::Color(220, 190, 255),
    sf::Color(170, 110, 40),
    sf::Color(255, 250, 200),
    sf::Color(128, 0,   0),
    sf::Color(170, 255, 195),
    sf::Color(128, 128, 0),
    sf::Color(255, 215, 180),
    sf::Color(0,   0,   128),
    sf::Color(128, 128, 128),
};

// ---------------------------------------------------------------------------
// Load a plan file:  t:(x0,y0),(x1,y1),...
// ---------------------------------------------------------------------------
static Solution load_plan(const std::string& path, const Instance& ins)
{
    const uint W = ins.G.width;
    std::ifstream f(path);
    if (!f.is_open()) return {};

    Solution sol;
    std::string line;
    const std::regex coord_re(R"(\((\d+),(\d+)\))");

    while (std::getline(f, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string tail = line.substr(colon + 1);

        Config cfg;
        auto it  = std::sregex_iterator(tail.begin(), tail.end(), coord_re);
        for (; it != std::sregex_iterator(); ++it) {
            int x = std::stoi((*it)[1]);
            int y = std::stoi((*it)[2]);
            uint idx = static_cast<uint>(y * W + x);
            if (idx >= ins.G.U.size() || ins.G.U[idx] == nullptr) return {};
            cfg.push_back(ins.G.U[idx]);
        }
        if (cfg.empty()) continue;
        if (!sol.empty() && cfg.size() != sol[0].size()) return {};
        sol.push_back(cfg);
    }
    return sol;
}

// ---------------------------------------------------------------------------
// Open file dialog via zenity
// ---------------------------------------------------------------------------
static std::string open_file_dialog()
{
    FILE* p = popen(
        "zenity --file-selection --title='Load Plan' "
        "--file-filter='Plan files (*.txt) | *.txt' "
        "--file-filter='All files | *' 2>/dev/null",
        "r");
    if (!p) return "";
    char buf[4096] = {};
    if (fgets(buf, sizeof(buf), p)) {
        std::string s(buf);
        if (!s.empty() && s.back() == '\n') s.pop_back();
        pclose(p);
        return s;
    }
    pclose(p);
    return "";
}

// ---------------------------------------------------------------------------
// Button helper
// ---------------------------------------------------------------------------
struct Button {
    sf::RectangleShape shape;
    sf::Text           label;
    bool               enabled = true;

    Button() = default;
    Button(float x, float y, float w, float h,
           const std::string& text, const sf::Font& font)
    {
        shape.setSize({w, h});
        shape.setPosition(x, y);
        shape.setOutlineColor(sf::Color(180, 180, 180));
        shape.setOutlineThickness(1.f);

        label.setFont(font);
        label.setString(text);
        label.setCharacterSize(14);
        label.setFillColor(sf::Color::White);
        sf::FloatRect tb = label.getLocalBounds();
        label.setOrigin(tb.left + tb.width / 2.f, tb.top + tb.height / 2.f);
        label.setPosition(x + w / 2.f, y + h / 2.f);
    }

    bool contains(sf::Vector2f p) const {
        return enabled && shape.getGlobalBounds().contains(p);
    }

    void updateColors(sf::Vector2f mouse) {
        if (!enabled) {
            shape.setFillColor(sf::Color(55, 55, 55));
            label.setFillColor(sf::Color(110, 110, 110));
        } else if (contains(mouse)) {
            shape.setFillColor(sf::Color(100, 160, 210));
            label.setFillColor(sf::Color::White);
        } else {
            shape.setFillColor(sf::Color(70, 130, 180));
            label.setFillColor(sf::Color::White);
        }
    }

    void draw(sf::RenderWindow& w) { w.draw(shape); w.draw(label); }
};

// ---------------------------------------------------------------------------
// Wrap a long string into multiple lines at max_chars width
// ---------------------------------------------------------------------------
static std::vector<std::string> wrap(const std::string& s, size_t max_chars)
{
    std::vector<std::string> lines;
    std::istringstream ss(s);
    std::string line;
    while (std::getline(ss, line)) {
        while (line.size() > max_chars) {
            lines.push_back(line.substr(0, max_chars));
            line = line.substr(max_chars);
        }
        lines.push_back(line);
    }
    return lines;
}

// ---------------------------------------------------------------------------
// run_gui
// ---------------------------------------------------------------------------
void run_gui(const Instance& ins)
{
    const auto& G = ins.G;
    const uint W  = G.width;
    const uint H  = G.height;

    // ---- layout ----
    const int CELL      = 20;
    const int MARGIN    = 20;
    const int PANEL_W   = 310;   // wide enough for feature text
    const int PANEL_PAD = 10;

    const int map_px = static_cast<int>(W) * CELL;
    const int win_w  = map_px + 2 * MARGIN + PANEL_W;
    const int win_h  = static_cast<int>(H) * CELL + 2 * MARGIN;

    sf::RenderWindow window(
        sf::VideoMode(static_cast<unsigned>(win_w),
                      static_cast<unsigned>(win_h)),
        "LaCAM2 Visualizer",
        sf::Style::Close | sf::Style::Titlebar);
    window.setFramerateLimit(60);

    // ---- font ----
    sf::Font font;
    const bool font_ok = font.loadFromFile(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    sf::Font mono;
    const bool mono_ok = mono.loadFromFile(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf");

    // ---- panel ----
    const float panel_x = static_cast<float>(map_px + 2 * MARGIN);
    sf::RectangleShape panel_bg(
        sf::Vector2f(static_cast<float>(PANEL_W), static_cast<float>(win_h)));
    panel_bg.setPosition(panel_x, 0.f);
    panel_bg.setFillColor(sf::Color(32, 32, 32));

    // ---- buttons ----
    const float BW = PANEL_W - 2 * PANEL_PAD;
    const float BH = 30.f;
    const float bx = panel_x + PANEL_PAD;
    Button btn_load, btn_step, btn_exit;
    if (font_ok) {
        float by = PANEL_PAD + 22.f;
        btn_load = Button(bx, by, BW, BH, "Load Plan", font); by += BH + 8.f;
        btn_step = Button(bx, by, BW, BH, "Step",      font);
        btn_exit = Button(bx, win_h - PANEL_PAD - BH, BW, BH, "Exit", font);
    }

    // ---- panel title ----
    sf::Text panel_title;
    if (font_ok) {
        panel_title.setFont(font);
        panel_title.setString("Controls");
        panel_title.setCharacterSize(13);
        panel_title.setStyle(sf::Text::Bold);
        panel_title.setFillColor(sf::Color(200, 200, 200));
        panel_title.setPosition(panel_x + PANEL_PAD, PANEL_PAD);
    }

    // ---- info + step texts (small status area below buttons) ----
    sf::Text info_text, step_text;
    const float info_y = PANEL_PAD + 22.f + 2 * (BH + 8.f) + 8.f;
    if (font_ok) {
        info_text.setFont(font);
        info_text.setCharacterSize(11);
        info_text.setFillColor(sf::Color(180, 220, 180));
        info_text.setPosition(panel_x + PANEL_PAD, info_y);

        step_text.setFont(font);
        step_text.setCharacterSize(11);
        step_text.setFillColor(sf::Color(200, 200, 200));
        step_text.setPosition(panel_x + PANEL_PAD, info_y + 30.f);
    }

    // ---- agent-detail area (scrollable monospace text) ----
    const float detail_y    = info_y + 60.f;
    // detail box ends just above the ring section
    const float ring_section_y_pre = static_cast<float>(win_h)
                                     - static_cast<float>(PANEL_PAD) - BH
                                     - 8.f - (16.f + 4*(26.f+4.f));
    const float detail_h    = ring_section_y_pre - 8.f - detail_y;
    const float DETAIL_LINE = 13.f;
    const int   DETAIL_CHARS = (PANEL_W - 2 * PANEL_PAD) / 7;

    sf::RectangleShape detail_bg(sf::Vector2f(BW, detail_h));
    detail_bg.setPosition(panel_x + PANEL_PAD, detail_y);
    detail_bg.setFillColor(sf::Color(24, 24, 24));

    // ---- grid tiles ----
    sf::VertexArray floor_va(sf::Quads), wall_va(sf::Quads);
    for (uint y = 0; y < H; ++y) {
        for (uint x = 0; x < W; ++x) {
            uint idx = y * W + x;
            bool pass = (G.U[idx] != nullptr);
            auto& va = pass ? floor_va : wall_va;
            sf::Color col = pass ? sf::Color(0, 0, 0) : sf::Color(255, 255, 255);
            float px = static_cast<float>(MARGIN + x * CELL);
            float py = static_cast<float>(MARGIN + y * CELL);
            va.append(sf::Vertex({px,        py       }, col));
            va.append(sf::Vertex({px + CELL, py       }, col));
            va.append(sf::Vertex({px + CELL, py + CELL}, col));
            va.append(sf::Vertex({px,        py + CELL}, col));
        }
    }

    // ---- agent / goal shapes ----
    const float AGENT_R   = CELL * 0.38f;
    const float GOAL_HALF = CELL * 0.22f;
    sf::CircleShape    circle(AGENT_R);    circle.setOrigin(AGENT_R, AGENT_R);
    sf::RectangleShape goal_rect({GOAL_HALF * 2.f, GOAL_HALF * 2.f});
    goal_rect.setOrigin(GOAL_HALF, GOAL_HALF);

    // Selection ring (around agent)
    sf::CircleShape sel_ring(AGENT_R + 3.f);
    sel_ring.setOrigin(AGENT_R + 3.f, AGENT_R + 3.f);
    sel_ring.setFillColor(sf::Color::Transparent);
    sel_ring.setOutlineColor(sf::Color::Yellow);
    sel_ring.setOutlineThickness(2.f);

    // Goal highlight ring (around selected agent's goal)
    const float GOAL_HL = GOAL_HALF + 4.f;
    sf::RectangleShape goal_hl({GOAL_HL * 2.f, GOAL_HL * 2.f});
    goal_hl.setOrigin(GOAL_HL, GOAL_HL);
    goal_hl.setFillColor(sf::Color::Transparent);
    goal_hl.setOutlineColor(sf::Color::Yellow);
    goal_hl.setOutlineThickness(2.f);

    // ---- ring button area (between detail box and Exit) ----
    const float BH_RING    = 26.f;
    const float RING_GAP   = 4.f;
    // Reserve space at bottom of panel: label + 4 buttons + exit
    const float ring_section_h = 16.f + 4 * (BH_RING + RING_GAP);
    const float ring_section_y = static_cast<float>(win_h)
                                 - static_cast<float>(PANEL_PAD) - BH
                                 - 8.f - ring_section_h;
    // Shrink the detail box to not overlap the ring section
    // (detail_h is calculated later; we patch it here via an adjusted value)
    // We define ring buttons now.
    std::array<Button, N_RINGS> btn_ring;
    sf::Text ring_section_label;
    if (font_ok) {
        ring_section_label.setFont(font);
        ring_section_label.setString("Radar Rings");
        ring_section_label.setCharacterSize(12);
        ring_section_label.setStyle(sf::Text::Bold);
        ring_section_label.setFillColor(sf::Color(160, 160, 200));
        ring_section_label.setPosition(panel_x + PANEL_PAD, ring_section_y);
        float ry = ring_section_y + 18.f;
        for (int r = 0; r < N_RINGS; ++r) {
            btn_ring[r] = Button(bx, ry, BW, BH_RING,
                                 std::string(RING_LABELS[r]), font);
            ry += BH_RING + RING_GAP;
        }
    }

    // ---- policy controller ----
    PolicyController policy(ins);

    // ---- plan / selection state ----
    Solution plan;
    int      plan_step      = -1;
    int      selected_agent = -1;   // -1 = none
    int      selected_ring  = -1;   // -1 = none
    int      detail_scroll  = 0;
    RingMap  ring_cells;
    bool     ring_cells_dirty = true;

    std::vector<std::string> detail_lines;

    auto current_config = [&]() -> const Config& {
        if (plan.empty() || plan_step < 0) return ins.starts;
        return plan[static_cast<size_t>(plan_step)];
    };

    // rebuild_detail: re-wrap feature text or ring detail for the detail pane.
    auto rebuild_detail = [&](bool keep_scroll = false) {
        if (!keep_scroll) detail_scroll = 0;
        detail_lines.clear();
        if (selected_agent < 0) {
            detail_lines.push_back("Click an agent to");
            detail_lines.push_back("inspect its features.");
            return;
        }
        const uint ai   = static_cast<uint>(selected_agent);
        const Vertex* v = current_config()[ai];
        int cx = static_cast<int>(v->index % W);
        int cy = static_cast<int>(v->index / W);
        const AgentFeatures& af = policy.context()[ai];
        std::string raw;
        if (selected_ring >= 0)
            raw = af.ring_detail(selected_ring);
        else
            raw = af.summary(ai, cx, cy);
        detail_lines = wrap(raw, static_cast<size_t>(DETAIL_CHARS));
    };

    auto refresh_features = [&](bool keep_scroll = false) {
        policy.compute(current_config(), plan_step);
        ring_cells_dirty = true;
        rebuild_detail(keep_scroll);
    };

    // Initial compute
    policy.compute(ins.starts, -1);
    rebuild_detail();

    // ---- hit-test: which agent is at pixel (mx,my)? ----
    auto agent_at = [&](sf::Vector2f mp) -> int {
        const Config& cfg = current_config();
        for (uint i = 0; i < static_cast<uint>(cfg.size()); ++i) {
            const Vertex* v = cfg[i];
            float ax = MARGIN + (v->index % W) * CELL + CELL * 0.5f;
            float ay = MARGIN + (v->index / W) * CELL + CELL * 0.5f;
            float dx = mp.x - ax, dy = mp.y - ay;
            if (std::sqrt(dx*dx + dy*dy) <= AGENT_R + 2.f)
                return static_cast<int>(i);
        }
        return -1;
    };

    // ---- main loop ----
    while (window.isOpen()) {
        sf::Vector2i mi = sf::Mouse::getPosition(window);
        sf::Vector2f mf(static_cast<float>(mi.x), static_cast<float>(mi.y));

        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Space) {
                    if (!plan.empty()) {
                        plan_step = (plan_step + 1) %
                                    static_cast<int>(plan.size());
                        refresh_features(/*keep_scroll=*/true);
                    }
                } else if (event.key.code == sf::Keyboard::Escape) {
                    selected_agent   = -1;
                    selected_ring    = -1;
                    ring_cells_dirty = true;
                    rebuild_detail();
                }
            }

            // Mouse-wheel scrolls the detail pane
            if (event.type == sf::Event::MouseWheelScrolled) {
                sf::FloatRect dr(panel_x + PANEL_PAD, detail_y, BW, detail_h);
                if (dr.contains(mf)) {
                    detail_scroll -= static_cast<int>(event.mouseWheelScroll.delta);
                    int max_scroll = std::max(0,
                        static_cast<int>(detail_lines.size()) -
                        static_cast<int>(detail_h / DETAIL_LINE));
                    detail_scroll = std::max(0, std::min(detail_scroll, max_scroll));
                }
            }

            if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left)
            {
                if (btn_exit.contains(mf)) {
                    window.close();
                }
                else if (btn_load.contains(mf)) {
                    std::string path = open_file_dialog();
                    if (!path.empty()) {
                        Solution loaded = load_plan(path, ins);
                        if (loaded.empty()) {
                            info_text.setString("Load failed.");
                            info_text.setFillColor(sf::Color(255, 100, 100));
                        } else {
                            plan      = std::move(loaded);
                            plan_step = 0;
                            auto sl   = path.rfind('/');
                            std::string fname = (sl == std::string::npos)
                                                ? path : path.substr(sl + 1);
                            info_text.setString(
                                fname + "\n" +
                                std::to_string(plan.size()) + " steps, " +
                                std::to_string(plan[0].size()) + " agents");
                            info_text.setFillColor(sf::Color(180, 220, 180));
                            policy.set_learning_mode(plan);
                            refresh_features();
                        }
                    }
                }
                else if (btn_step.contains(mf)) {
                    if (!plan.empty()) {
                        plan_step = (plan_step + 1) %
                                    static_cast<int>(plan.size());
                        refresh_features(/*keep_scroll=*/true);
                    }
                }
                else {
                    // Click on map — try to select an agent
                    int hit = agent_at(mf);
                    if (hit >= 0) {
                        selected_agent = hit;
                        selected_ring  = -1;
                        ring_cells_dirty = true;
                        refresh_features();
                    }
                    // Check ring buttons
                    for (int r = 0; r < N_RINGS; ++r) {
                        if (btn_ring[r].contains(mf)) {
                            selected_ring = (selected_ring == r) ? -1 : r;
                            ring_cells_dirty = true;
                            rebuild_detail();
                            break;
                        }
                    }
                }
            }
        }

        // ---- update button states ----
        btn_step.enabled = !plan.empty();
        for (int r = 0; r < N_RINGS; ++r)
            btn_ring[r].enabled = (selected_agent >= 0);
        btn_load.updateColors(mf);
        btn_step.updateColors(mf);
        btn_exit.updateColors(mf);
        for (int r = 0; r < N_RINGS; ++r) {
            if (!btn_ring[r].enabled) {
                btn_ring[r].shape.setFillColor(sf::Color(45, 45, 45));
                btn_ring[r].label.setFillColor(sf::Color(90, 90, 90));
            } else if (r == selected_ring) {
                btn_ring[r].shape.setFillColor(sf::Color(40, 120, 100));
                btn_ring[r].label.setFillColor(sf::Color::White);
            } else if (btn_ring[r].contains(mf)) {
                btn_ring[r].shape.setFillColor(sf::Color(70, 150, 130));
                btn_ring[r].label.setFillColor(sf::Color::White);
            } else {
                btn_ring[r].shape.setFillColor(sf::Color(55, 90, 80));
                btn_ring[r].label.setFillColor(sf::Color(180, 220, 210));
            }
        }

        if (font_ok) {
            if (plan_step >= 0) {
                std::string mode_str = (policy.mode() == PolicyMode::Learning)
                                       ? " [Learning]" : " [Execution]";
                step_text.setString("Step: " + std::to_string(plan_step) +
                                    " / " + std::to_string((int)plan.size()-1) +
                                    mode_str);
            } else {
                step_text.setString("Space=Step  Esc=Deselect");
                step_text.setFillColor(sf::Color(130, 130, 130));
            }
        }

        // ---- render ----
        window.clear(sf::Color(20, 20, 20));
        window.draw(floor_va);
        window.draw(wall_va);

        // ring highlight (drawn before agents so agents appear on top)
        if (selected_agent >= 0 && selected_ring >= 0) {
            if (ring_cells_dirty) {
                ring_cells = policy.compute_ring_map(
                    static_cast<uint>(selected_agent), current_config());
                ring_cells_dirty = false;
            }
            // Two alternating grays for even/odd sectors
            static const sf::Color SECTOR_COL[2] = {
                sf::Color(65, 65, 65),
                sf::Color(95, 95, 95)
            };
            sf::VertexArray ring_va(sf::Quads);
            for (int sec = 0; sec < N_ANGLES; ++sec) {
                sf::Color col = SECTOR_COL[sec % 2];
                for (uint vi : ring_cells[selected_ring][sec]) {
                    float px = static_cast<float>(MARGIN + (vi % W) * CELL);
                    float py = static_cast<float>(MARGIN + (vi / W) * CELL);
                    ring_va.append(sf::Vertex({px,        py       }, col));
                    ring_va.append(sf::Vertex({px + CELL, py       }, col));
                    ring_va.append(sf::Vertex({px + CELL, py + CELL}, col));
                    ring_va.append(sf::Vertex({px,        py + CELL}, col));
                }
            }
            window.draw(ring_va);
        }

        // goals
        for (uint i = 0; i < ins.N; ++i) {
            sf::Color col = PALETTE[i % PALETTE.size()]; col.a = 160;
            goal_rect.setFillColor(col);
            const Vertex* gv = ins.goals[i];
            float gpx = MARGIN + (gv->index % W) * CELL + CELL * 0.5f;
            float gpy = MARGIN + (gv->index / W) * CELL + CELL * 0.5f;
            goal_rect.setPosition(gpx, gpy);
            window.draw(goal_rect);
            // highlight the selected agent's goal
            if (static_cast<int>(i) == selected_agent) {
                goal_hl.setPosition(gpx, gpy);
                window.draw(goal_hl);
            }
        }

        // agents
        const Config& cfg = current_config();
        for (uint i = 0; i < static_cast<uint>(cfg.size()); ++i) {
            const Vertex* v = cfg[i];
            float ax = MARGIN + (v->index % W) * CELL + CELL * 0.5f;
            float ay = MARGIN + (v->index / W) * CELL + CELL * 0.5f;

            if (static_cast<int>(i) == selected_agent) {
                sel_ring.setPosition(ax, ay);
                window.draw(sel_ring);
            }

            circle.setFillColor(PALETTE[i % PALETTE.size()]);
            circle.setPosition(ax, ay);
            window.draw(circle);

            // compass arrow for non-zero intent
            const AgentFeatures& af = policy.context()[i];
            float ix = af.intent_x, iy = af.intent_y;
            float imag = std::sqrt(ix*ix + iy*iy);
            if (imag > 1e-4f) {
                float shaft = AGENT_R * 0.72f;
                float tx = ax + ix * shaft;
                float ty = ay + iy * shaft;
                // perpendicular for arrowhead wings
                float px2 = -iy / imag, py2 = ix / imag;
                float hw = AGENT_R * 0.22f;  // half-width of head
                float hb = AGENT_R * 0.28f;  // how far back the base is
                sf::VertexArray arrow(sf::Lines, 2);
                arrow[0] = sf::Vertex({ax, ay}, sf::Color::White);
                arrow[1] = sf::Vertex({tx, ty}, sf::Color::White);
                window.draw(arrow);
                sf::VertexArray head(sf::Triangles, 3);
                head[0] = sf::Vertex({tx + ix/imag * AGENT_R * 0.2f, ty + iy/imag * AGENT_R * 0.2f}, sf::Color::White);
                head[1] = sf::Vertex({tx - ix/imag*hb + px2*hw, ty - iy/imag*hb + py2*hw}, sf::Color::White);
                head[2] = sf::Vertex({tx - ix/imag*hb - px2*hw, ty - iy/imag*hb - py2*hw}, sf::Color::White);
                window.draw(head);
            }
        }

        // panel background
        window.draw(panel_bg);

        // buttons / status
        btn_load.draw(window);
        btn_step.draw(window);
        btn_exit.draw(window);
        if (font_ok) {
            window.draw(panel_title);
            window.draw(info_text);
            window.draw(step_text);
            window.draw(ring_section_label);
        }
        for (int r = 0; r < N_RINGS; ++r) btn_ring[r].draw(window);

        // feature detail pane
        window.draw(detail_bg);
        if ((font_ok || mono_ok) && !detail_lines.empty()) {
            sf::Text line_text;
            line_text.setFont(mono_ok ? mono : font);
            line_text.setCharacterSize(11);
            // dim colour for hint (no agent selected), bright for features
            line_text.setFillColor(selected_agent < 0
                ? sf::Color(90, 90, 90)
                : sf::Color(210, 230, 210));

            int visible = static_cast<int>(detail_h / DETAIL_LINE);
            int start   = detail_scroll;
            int end_idx = std::min(start + visible,
                                   static_cast<int>(detail_lines.size()));

            // Scissor via View is not trivial in SFML 2; just clip by not
            // drawing lines that would fall outside the box.
            for (int li = start; li < end_idx; ++li) {
                float ty = detail_y + (li - start) * DETAIL_LINE;
                if (ty + DETAIL_LINE > detail_y + detail_h) break;
                line_text.setString(detail_lines[li]);
                line_text.setPosition(panel_x + PANEL_PAD, ty);
                window.draw(line_text);
            }
        }

        window.display();
    }
}
