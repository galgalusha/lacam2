#pragma once

#include <functional>
#include <memory>
#include "dist_table.hpp"
#include "graph.hpp"

// low-level node
struct LNode {
  std::shared_ptr<LNode> parent;
  uint _who;
  Vertex* _where;
  const uint depth;
  LNode(std::shared_ptr<LNode> parent = nullptr, uint i = 0,
        Vertex* v = nullptr);  // who and where
  virtual std::vector<uint> who() const;
  virtual Vertices where() const;
  virtual ~LNode() = default;
};

// high-level node
struct HNode {
  static uint HNODE_CNT;  // count #(high-level node)
  const Config C;
  int depth;

  // tree
  HNode* parent;
  std::set<HNode*> neighbor;

  // costs
  uint g;        // g-value (might be updated)
  uint h;        // h-value (might be updated)
  uint f;        // g + h (might be updated)

  // for low-level search
  std::vector<float> priorities;
  std::vector<uint> order;
  std::queue<std::shared_ptr<LNode>> search_tree;
  uint ll_search;
  bool max_ll_already_decayed;
  float max_ll;

  HNode(const Config& _C, DistTable& D, HNode* _parent, const uint _g,
        const uint _h);
  ~HNode();
  void initialize_order(DistTable& D);
};
using HNodes = std::vector<HNode*>;
