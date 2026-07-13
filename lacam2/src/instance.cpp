#include "../include/instance.hpp"

#include <stdexcept>

Instance::Instance(const std::string& map_filename,
                   const std::vector<uint>& start_indexes,
                   const std::vector<uint>& goal_indexes)
    : G_owned(std::make_unique<Graph>(map_filename)),
      G(*G_owned),
      starts(),
      goals(),
      N(start_indexes.size())
{
  for (auto k : start_indexes) starts.push_back(G.U[k]);
  for (auto k : goal_indexes)  goals.push_back(G.U[k]);
}

Instance::Instance(const std::vector<std::string>& grid,
                   const std::vector<uint>& start_indexes,
                   const std::vector<uint>& goal_indexes)
    : G_owned(std::make_unique<Graph>(grid)),
      G(*G_owned),
      starts(),
      goals(),
      N(start_indexes.size())
{
  for (auto k : start_indexes) starts.push_back(G.U[k]);
  for (auto k : goal_indexes)  goals.push_back(G.U[k]);
}

// for load instance
static const std::regex r_instance =
    std::regex(R"(\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+)");

Instance::Instance(const std::string& scen_filename,
                   const std::string& map_filename, const uint _N)
    : G_owned(std::make_unique<Graph>(map_filename)),
      G(*G_owned),
      starts(),
      goals(),
      N(_N)
{
  // load start-goal pairs
  std::ifstream file(scen_filename);
  if (!file) {
    info(0, 0, scen_filename, " is not found");
    return;
  }
  std::string line;
  std::smatch results;

  while (getline(file, line)) {
    // for CRLF coding
    if (*(line.end() - 1) == 0x0d) line.pop_back();

    if (std::regex_match(line, results, r_instance)) {
      uint x_s = std::stoi(results[1].str());
      uint y_s = std::stoi(results[2].str());
      uint x_g = std::stoi(results[3].str());
      uint y_g = std::stoi(results[4].str());
      if (x_s < 0 || G.width <= x_s || x_g < 0 || G.width <= x_g) break;
      if (y_s < 0 || G.height <= y_s || y_g < 0 || G.height <= y_g) break;
      auto s = G.U[G.width * y_s + x_s];
      auto g = G.U[G.width * y_g + x_g];
      if (s == nullptr || g == nullptr) break;
      starts.push_back(s);
      goals.push_back(g);
    }

    if (starts.size() == N) break;
  }
}

Instance::Instance(const std::string& map_filename, std::mt19937* MT,
                   const uint _N)
    : G_owned(std::make_unique<Graph>(map_filename)),
      G(*G_owned),
      starts(),
      goals(),
      N(_N)
{
  // random assignment
  const auto V_size = G.size();

  // set starts
  auto s_indexes = std::vector<uint>(V_size);
  std::iota(s_indexes.begin(), s_indexes.end(), 0);
  std::shuffle(s_indexes.begin(), s_indexes.end(), *MT);
  size_t i = 0;
  while (true) {
    if (i >= V_size) return;
    starts.push_back(G.V[s_indexes[i]]);
    if (starts.size() == N) break;
    ++i;
  }

  // set goals
  auto g_indexes = std::vector<uint>(V_size);
  std::iota(g_indexes.begin(), g_indexes.end(), 0);
  std::shuffle(g_indexes.begin(), g_indexes.end(), *MT);
  size_t j = 0;
  while (true) {
    if (j >= V_size) return;
    goals.push_back(G.V[g_indexes[j]]);
    if (goals.size() == N) break;
    ++j;
  }
}

Instance::Instance(const Instance& parent, const Config& new_starts, const Config& new_goals)
    : G_owned(nullptr),
      G(parent.G),
      starts(new_starts),
      goals(new_goals),
      N(new_starts.size())
{
  // Borrows parent.G — vertex pointers in starts/goals are already from parent.G.
}

Instance Instance::multiply(int n) const
{
  if (n <= 0) throw std::invalid_argument("multiply expects n >= 1");
  if (starts.size() != goals.size()) {
    throw std::runtime_error("invalid instance: starts/goals size mismatch");
  }

  const uint base_w = G.width;
  const uint base_h = G.height;
  const uint tiled_w = static_cast<uint>(n) * base_w + static_cast<uint>(n - 1);
  const uint agent_count = starts.size();

  // rebuild the base occupancy grid from Graph::U
  std::vector<std::string> tiled_grid(base_h, std::string(tiled_w, '@'));
  for (uint y = 0; y < base_h; ++y) {
    for (uint x = 0; x < base_w; ++x) {
      if (G.U[base_w * y + x] != nullptr) tiled_grid[y][x] = '.';
    }
  }

  // copy the base map into each sub-grid and keep separator columns as walls
  for (int tile = 1; tile < n; ++tile) {
    const uint offset_x = static_cast<uint>(tile) * (base_w + 1);
    for (uint y = 0; y < base_h; ++y) {
      for (uint x = 0; x < base_w; ++x) {
        if (G.U[base_w * y + x] != nullptr) tiled_grid[y][offset_x + x] = '.';
      }
    }
  }

  std::vector<uint> start_indexes;
  std::vector<uint> goal_indexes;
  start_indexes.reserve(agent_count * static_cast<uint>(n));
  goal_indexes.reserve(agent_count * static_cast<uint>(n));

  for (int tile = 0; tile < n; ++tile) {
    const uint offset_x = static_cast<uint>(tile) * (base_w + 1);
    for (uint i = 0; i < agent_count; ++i) {
      const uint s_index = starts[i]->index;
      const uint g_index = goals[i]->index;

      const uint s_x = s_index % base_w;
      const uint s_y = s_index / base_w;
      const uint g_x = g_index % base_w;
      const uint g_y = g_index / base_w;

      start_indexes.push_back(tiled_w * s_y + (offset_x + s_x));
      goal_indexes.push_back(tiled_w * g_y + (offset_x + g_x));
    }
  }

  return Instance(tiled_grid, start_indexes, goal_indexes);
}

void Instance::render(std::ostream& os) const
{
  std::vector<std::string> canvas(G.height, std::string(G.width, '#'));

  for (uint y = 0; y < G.height; ++y) {
    for (uint x = 0; x < G.width; ++x) {
      if (G.U[G.width * y + x] != nullptr) canvas[y][x] = '.';
    }
  }

  for (uint agent_id = 0; agent_id < starts.size(); ++agent_id) {
    auto v = starts[agent_id];
    if (v == nullptr) continue;
    const uint idx = v->index;
    const uint x = idx % G.width;
    const uint y = idx / G.width;
    canvas[y][x] = static_cast<char>('0' + (agent_id % 10));
  }

  for (const auto& row : canvas) os << row << '\n';
}

bool Instance::is_valid(const int verbose) const
{
  if (N != starts.size() || N != goals.size()) {
    info(1, verbose, "invalid N, check instance");
    return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const Solution& solution)
{
  auto N = solution.front().size();
  for (size_t i = 0; i < N; ++i) {
    os << std::setw(5) << i << ":";
    for (size_t k = 0; k < solution[i].size(); ++k) {
      if (k > 0) os << "->";
      os << std::setw(5) << solution[i][k]->index;
    }
    os << std::endl;
  }
  return os;
}
