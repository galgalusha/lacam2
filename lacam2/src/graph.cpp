#include "../include/graph.hpp"

static void build_edges(Graph& g);

Vertex::Vertex(uint _id, uint _index)
    : id(_id), index(_index), neighbor(Vertices())
{
}

Graph::Graph() : V(Vertices()), width(0), height(0) {}
Graph::~Graph()
{
  for (auto& v : V)
    if (v != nullptr) delete v;
  V.clear();
}

// to load graph
static const std::regex r_height = std::regex(R"(height\s(\d+))");
static const std::regex r_width = std::regex(R"(width\s(\d+))");
static const std::regex r_map = std::regex(R"(map)");

Graph::Graph(const std::string& filename) : V(Vertices()), width(0), height(0)
{
  std::ifstream file(filename);
  if (!file) {
    std::cout << "file " << filename << " is not found." << std::endl;
    return;
  }
  std::string line;
  std::smatch results;

  // read fundamental graph parameters
  while (getline(file, line)) {
    // for CRLF coding
    if (*(line.end() - 1) == 0x0d) line.pop_back();

    if (std::regex_match(line, results, r_height)) {
      height = std::stoi(results[1].str());
    }
    if (std::regex_match(line, results, r_width)) {
      width = std::stoi(results[1].str());
    }
    if (std::regex_match(line, results, r_map)) break;
  }

  U = Vertices(width * height, nullptr);

  // create vertices
  uint y = 0;
  while (getline(file, line)) {
    // for CRLF coding
    if (*(line.end() - 1) == 0x0d) line.pop_back();
    for (uint x = 0; x < width; ++x) {
      char s = line[x];
      if (s == 'T' or s == '@') continue;  // object
      auto index = width * y + x;
      auto v = new Vertex(V.size(), index);
      V.push_back(v);
      U[index] = v;
    }
    ++y;
  }
  file.close();

  // create edges
  build_edges(*this);
}

uint Graph::size() const { return V.size(); }

// shared helper: build neighbour edges once U/width/height are populated
static void build_edges(Graph& g)
{
  for (uint y = 0; y < g.height; ++y) {
    for (uint x = 0; x < g.width; ++x) {
      auto v = g.U[g.width * y + x];
      if (v == nullptr) continue;
      if (x > 0) {
        auto u = g.U[g.width * y + (x - 1)];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      if (x < g.width - 1) {
        auto u = g.U[g.width * y + (x + 1)];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      if (y < g.height - 1) {
        auto u = g.U[g.width * (y + 1) + x];
        if (u != nullptr) v->neighbor.push_back(u);
      }
      if (y > 0) {
        auto u = g.U[g.width * (y - 1) + x];
        if (u != nullptr) v->neighbor.push_back(u);
      }
    }
  }
}

Graph::Graph(const std::vector<std::string>& grid)
    : V(Vertices()), width(0), height(0)
{
  if (grid.empty()) return;
  height = grid.size();
  width = grid[0].size();
  U = Vertices(width * height, nullptr);

  for (uint y = 0; y < height; ++y) {
    for (uint x = 0; x < width; ++x) {
      if (grid[y][x] != '.') continue;
      auto index = width * y + x;
      auto v = new Vertex(V.size(), index);
      V.push_back(v);
      U[index] = v;
    }
  }

  build_edges(*this);
}

bool is_same_config(const Config& C1, const Config& C2)
{
  const auto N = C1.size();
  for (size_t i = 0; i < N; ++i) {
    if (C1[i]->id != C2[i]->id) return false;
  }
  return true;
}

uint ConfigHasher::operator()(const Config& C) const
{
  uint hash = C.size();
  for (auto& v : C) hash ^= v->id + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  return hash;
}

std::ostream& operator<<(std::ostream& os, const Vertex* v)
{
  os << v->index;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Config& config)
{
  os << "<";
  const auto N = config.size();
  for (size_t i = 0; i < N; ++i) {
    if (i > 0) os << ",";
    os << std::setw(5) << config[i];
  }
  os << ">";
  return os;
}
