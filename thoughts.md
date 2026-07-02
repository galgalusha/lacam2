# Objective: Implement Continuous CEM for Execution Priority

We are upgrading the Tie-Breaking Policy architecture to also learn and dictate the execution order (priority) of agents at each vertex. Because execution priority represents a position in a sequence, we must treat it as a continuous variable rather than a discrete categorical choice. We will use a Gaussian (Normal) distribution for priority.

Implement the following architectural and mathematical changes to handle the initialization and tracking of priority scores.

## 1. Data Structure Updates

**Target:** `policy.hpp` (or wherever policy structs are defined)

1. **Define the Gaussian Parameters:** Create a new struct to hold the mean and standard deviation for the priority distribution.
   ```cpp
   struct PriorityDist {
     double mu;      // The mean execution order
     double sigma;   // The exploration noise (standard deviation)
   };