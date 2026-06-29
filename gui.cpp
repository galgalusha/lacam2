#include "gui.hpp"

#include <SFML/Graphics.hpp>

#include <array>
#include <cmath>

// Fixed palette – colours are reused cyclically
static const std::array<sf::Color, 20> PALETTE = {
    sf::Color(230, 25,  75),   // red
    sf::Color(60,  180, 75),   // green
    sf::Color(255, 225, 25),   // yellow
    sf::Color(0,   130, 200),  // blue
    sf::Color(245, 130, 48),   // orange
    sf::Color(145, 30,  180),  // purple
    sf::Color(70,  240, 240),  // cyan
    sf::Color(240, 50,  230),  // magenta
    sf::Color(210, 245, 60),   // lime
    sf::Color(250, 190, 212),  // pink
    sf::Color(0,   128, 128),  // teal
    sf::Color(220, 190, 255),  // lavender
    sf::Color(170, 110, 40),   // brown
    sf::Color(255, 250, 200),  // beige
    sf::Color(128, 0,   0),    // maroon
    sf::Color(170, 255, 195),  // mint
    sf::Color(128, 128, 0),    // olive
    sf::Color(255, 215, 180),  // apricot
    sf::Color(0,   0,   128),  // navy
    sf::Color(128, 128, 128),  // grey
};

void run_gui(const Instance& ins)
{
    const auto& G = ins.G;
    const uint W = G.width;
    const uint H = G.height;

    // ---- layout constants ----
    const int CELL       = 20;    // pixels per grid cell
    const int MARGIN     = 40;    // pixels around the grid
    const int BOTTOM_BAR = 50;    // space for button at the bottom

    const int win_w = static_cast<int>(W) * CELL + 2 * MARGIN;
    const int win_h = static_cast<int>(H) * CELL + 2 * MARGIN + BOTTOM_BAR;

    sf::RenderWindow window(
        sf::VideoMode(static_cast<unsigned>(win_w),
                      static_cast<unsigned>(win_h)),
        "LaCAM2 Visualizer",
        sf::Style::Close | sf::Style::Titlebar);
    window.setFramerateLimit(60);

    // ---- font for button label ----
    sf::Font font;
    const bool font_ok = font.loadFromFile(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");

    // ---- pre-build static vertex arrays ----
    sf::VertexArray floor_va(sf::Quads);
    sf::VertexArray wall_va(sf::Quads);

    const sf::Color FLOOR_COL(240, 240, 235);
    const sf::Color WALL_COL (60,  60,  60);

    for (uint y = 0; y < H; ++y) {
        for (uint x = 0; x < W; ++x) {
            uint idx = y * W + x;
            bool passable = (G.U[idx] != nullptr);
            auto& va  = passable ? floor_va : wall_va;
            sf::Color col = passable ? FLOOR_COL : WALL_COL;

            float px = static_cast<float>(MARGIN + x * CELL);
            float py = static_cast<float>(MARGIN + y * CELL);

            va.append(sf::Vertex(sf::Vector2f(px,        py       ), col));
            va.append(sf::Vertex(sf::Vector2f(px + CELL, py       ), col));
            va.append(sf::Vertex(sf::Vector2f(px + CELL, py + CELL), col));
            va.append(sf::Vertex(sf::Vector2f(px,        py + CELL), col));
        }
    }

    // ---- Exit button ----
    const float BTN_W = 80.f, BTN_H = 30.f;
    const float btn_x = (win_w - BTN_W) / 2.f;
    const float btn_y = win_h - BOTTOM_BAR + (BOTTOM_BAR - BTN_H) / 2.f;

    sf::RectangleShape btn_shape(sf::Vector2f(BTN_W, BTN_H));
    btn_shape.setFillColor(sf::Color(70, 130, 180));
    btn_shape.setOutlineColor(sf::Color::White);
    btn_shape.setOutlineThickness(1.f);
    btn_shape.setPosition(btn_x, btn_y);

    sf::Text btn_label;
    if (font_ok) {
        btn_label.setFont(font);
        btn_label.setString("Exit");
        btn_label.setCharacterSize(16);
        btn_label.setFillColor(sf::Color::White);
        sf::FloatRect tb = btn_label.getLocalBounds();
        btn_label.setOrigin(tb.left + tb.width / 2.f, tb.top + tb.height / 2.f);
        btn_label.setPosition(btn_x + BTN_W / 2.f, btn_y + BTN_H / 2.f);
    }

    // ---- shape templates ----
    const float AGENT_R   = CELL * 0.38f;
    const float GOAL_HALF = CELL * 0.22f;

    sf::CircleShape circle(AGENT_R);
    circle.setOrigin(AGENT_R, AGENT_R);

    sf::RectangleShape goal_rect(sf::Vector2f(GOAL_HALF * 2.f, GOAL_HALF * 2.f));
    goal_rect.setOrigin(GOAL_HALF, GOAL_HALF);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                sf::Vector2f mp(static_cast<float>(event.mouseButton.x),
                                static_cast<float>(event.mouseButton.y));
                if (btn_shape.getGlobalBounds().contains(mp))
                    window.close();
            }
        }

        // Button hover highlight
        sf::Vector2i mpos = sf::Mouse::getPosition(window);
        sf::Vector2f mposf(static_cast<float>(mpos.x),
                           static_cast<float>(mpos.y));
        bool hovering = btn_shape.getGlobalBounds().contains(mposf);
        btn_shape.setFillColor(hovering ? sf::Color(100, 160, 210)
                                        : sf::Color(70, 130, 180));

        window.clear(sf::Color(30, 30, 30));
        window.draw(floor_va);
        window.draw(wall_va);

        // Draw goals and agents
        for (uint i = 0; i < ins.N; ++i) {
            const sf::Color col = PALETTE[i % PALETTE.size()];

            // goal (small rectangle)
            const Vertex* gv = ins.goals[i];
            uint gx = gv->index % W;
            uint gy = gv->index / W;
            sf::Color gc = col; gc.a = 180;
            goal_rect.setFillColor(gc);
            goal_rect.setPosition(
                MARGIN + gx * CELL + CELL * 0.5f,
                MARGIN + gy * CELL + CELL * 0.5f);
            window.draw(goal_rect);

            // agent (filled circle)
            const Vertex* sv = ins.starts[i];
            uint sx = sv->index % W;
            uint sy = sv->index / W;
            circle.setFillColor(col);
            circle.setPosition(
                MARGIN + sx * CELL + CELL * 0.5f,
                MARGIN + sy * CELL + CELL * 0.5f);
            window.draw(circle);
        }

        window.draw(btn_shape);
        if (font_ok) window.draw(btn_label);
        window.display();
    }
}

