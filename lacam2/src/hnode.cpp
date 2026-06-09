#include "../include/planner.hpp"

#include <algorithm>
#include <numeric>

uint HNode::HNODE_CNT = 0;

// for high-level
HNode::HNode(const Config& _C, DistTable& D, HNode* _parent, const uint _g,
             const uint _h)
    : C(_C),
      parent(_parent),
      neighbor(),
      g(_g),
      h(_h),
      f(g + h),
      priorities(C.size()),
      order(C.size(), 0),
      search_tree(std::queue<std::shared_ptr<LNode>>()),
      ll_search(0),
      max_ll_already_decayed(false),
      max_ll(Planner::max_ll)
{
  if (parent != nullptr) {
    depth = parent->depth + 1;
  } else {
    depth = 0;
  }

  ++HNODE_CNT;

  auto root = std::make_shared<LNode>();
  search_tree.push(root);
  ll_search += 1;

  initialize_order(D);
}

HNode::~HNode()
{
}

void HNode::initialize_order(DistTable& D)
{
  const auto N = C.size();

  // set priorities (original logic)
  if (parent == nullptr) {
    // initialize
    for (uint i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
  } else {
    // dynamic priorities, akin to PIBT
    for (size_t i = 0; i < N; ++i) {
      if (D.get(i, C[i]) != 0) {
        priorities[i] = parent->priorities[i] + 1;
      } else {
        priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
      }
    }
  }

  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](uint i, uint j) {
    return priorities[i] > priorities[j];
  });
}