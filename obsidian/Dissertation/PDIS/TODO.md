
* Introduction
	* Context
	
	* Motivation
		
	* Goals
		
	* Report Structure


	* Questions 
		* **Correct Structure**?
		
	* TODO
		* Everything

* Literature Review
	* Methodology
		* Research Questions
			* Can GRL algorithms be used to improve sequential decision making processes with graph-based environments?
				* How can RL algorithms be adapted to effectively learn policies on graph-structured data?
				* What are the most efficient ways to incorporate graph topology into the state or reward function of a reinforcement learning model?
					"Graph Reinforcement Learning" OR "Reinforcement Learning on Graphs"
					"Graph Convolutinal Network" AND "Reinforcement Learning"
					"Graph Attention Network" AND "Reinforcement Learning"
			 * DED
				 * **Can GRL algorithms provide robust solutions to the DED problem under different grid conditions and contingencies?**
				 * "Dispatch" and "Energy" and "Reinforcement Learning"
		* Search databases
			* Web of Science
			* Scopus
			* Google Scholar
		* Requirements
			* Less than 5 years or seminal
		
	* Dynamic Economic Dispatch
		* 3 approaches mentioned
		* 1 is actually real optimal dispatch


	* Questions
		* DED Research question
		* Show word map in methodology
		* Literature Review in DED

	* TODO
		* Methodology
		* DED section


* Problem Statement
	* Problem
		* Graph Reinforcement Learning -> Ok
		* Dynamic Economic Dispatch 
			* Objective Function
			* Constraints
		* GRL on DED
			* Graph Rerpresentation
			* MDP Reformulation
				* State Space: variables + G(Adj,X)
				* Action Space: change of generation output ΔPG(t), output of RES PRES(t), charging/discharging power of ESS PESS(t).
				* Reward Function: - 
	* **Scope** ?


	* Questions 
		* Scope Section
		* DED formulation

	* TODO
		* Finish DED formulation

* Problem and Proposed Solution
	* Problem Statement
		* 5W2H
	* Functional Requirements
	* Non-functional Requirements
	* Solution Architecture
		* DRL + GNN
	* Methodology
		* Train the GRL model with a simulation of a power distribution grid based on a IEEE Test Feeder 
		* Main elements modelled
			* Generators
			* Powerlines
			* Loads
			* Storage Units
		* Data provided by IEEE Power and Energy Society and simulation managed by the Grid2Op framework
	* Evaluative Methods
		* Training 
		* Daily Operating Cost
		* Scalability and Robustness
		* Computational Efficiency 
		* Compare SAC, DDPG and PPO with combination of GAT, GCN and GIN
	* Work Plan
		* Highlight and explain different phases

	* Questions
		* Graph representation
		* Evaluative Methods

	* TODO
		* Requirements
		* Solution Architecture
		* Methodology
		* 
* Conclusions
	* Contributions 
		
	* SWOT
		* Strengths:
			* Well-Formulated Problem
			* Well Maintaned Tech Stack
		* Weaknesses:
			* 
		* Opportunities:
			* Organize knowledge and compare different GRL approaches
		* Threats
			* Computational Resources
	* SMART 
		* Perform SMART Analysis

Questions
* Research Question for DED
* Which constraints and objective function for DED
* Which data is going to be used with which sizes and modifications
* MDP reformulation
	* We define the action at time step t as $a_t =\{\Delta P^{DG_d}_t, \Delta V^{DG_d}_t \}^D_{d=1}$, which are the output power of DGs. The action space $A \in ⋃^D_{d=1} [P^{DG_d}_\text{min}, P^{DG_d}_\text{max}, ]$ .
	* state space
		* $s_i(t) = \{P^G_i, Q^G_i, \overline{P}^\text{RES}_i, V^G_i, E^\text{ESS}_i\}$ 