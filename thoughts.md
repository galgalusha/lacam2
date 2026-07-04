# CEM and PolicyPIBT
This vscode project contains multiple planners, but to our context,
we only focus on CEMPlanner, which uses Cross-Entropy Method (CEM) to
optimize a probability model stored in ProbabilityPolicy to guide a PIBT
variance called PolicyPIBT. More specifically, PoicyPIBT uses a Policy interface
to ask for tie breaking using method get_neighbor_scores, which is called when
an agent located at vertex v, needs to choose between several neighbors of v, all
having equal D value (D = distance to goal).

# The master model (ProbabilityPolicy)
We refer ProbabilityPolicy as the "master" model.
It just contains a vector of AgentProbabilityPolicy, one model per agent.
AgentProbabilityPolicy contains ```unordered_map<Vertex*, std::unordered_map<Vertex*, double>> vertex_probs```,
which for any vertex v, holds a probability to choose any neighbor of v. The inner map is the 
prabability distribution of the neighbors.

WIP: there is a work in progress to add priority probability, priority_dist. Its currently only parially implemented
in some of the data structures.

# Initial Policy
CEMSolver::solve begins by running thousands of PIBT rollouts on a vanilla PIBT (not PolicyPIBT), which
uses a uniform probability for tie breaking. It filters the best rollouts by smallest SoC (cost metric)
and count the neighbor decisions made per agent per vertex and hold it in AgentScores.
The result is a vector of AgentScores stored in ScorePolicy.
Then it calls a function to_probability_policy with ScorePolicy to create the master model.
The rest of the execution of the planner will modify the parameters in the master model.
AgentScores and ScorePolicy only used for initialization and are forgotten afterwards.

# CEM iterations (generations)
The loop uses a randomizer to randomize deterministic tie breaking values per agent
stored in DeterministicPolicy. We have as many deterministic policies as number
of rollouts. We run them all in threads (parallel) and filter the best ones.
Then we take the best rollouts to create ScorePolicy (which is collected scores) 
and update the master policy.

# To sum up
## create initial policy:
* run rollouts
* filter best rollouts
* creates ScorePolicy from best rollouts
* creates master model from ScorePolicy
## CEM gen:
* randomize descrete policies
* run rollouts
* filter best rollouts
* creates ScorePolicy from best rollouts
* update master model from ScorePolicy

