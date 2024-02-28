
$\mathbb{E}$ - expectation operator
$T$ - Set of dispatch stages
$N^\text{BUS}$ - Set of buses
$N^G, N^\text{RES}, N^\text{ESS}$ - Set of conventional generation, RES and ESS
$N^G_i, N^\text{RES}_i, N^\text{ESS}_i$ - Set of conventional generation, RES and ESS connected to node i
$\Delta t$ - time interval
$a_i, b_i, c_i$ - Cost coefficient of conventional generation $i$
$\beta_\text{RES}$ - Penalty coefficient for RES Curtailment
$\beta_\text{ESS}$ - Operation cost coefficient for RES Curtailment
$\overline{P}^\text{RES}_i(t)$  - Maximum active power output of RES $i$
$\overline{P}^G_i , \underline{P}^G_i$ - Maximum / minimum power output of conventional generation $i$
$\overline{u}_i, \underline{u}_i$ - Maximum / minimum voltage magnitude of bus $i$
$G_{ij}, B_{ij}$ - Susceptance and conductance between buses $i$ and $j$
$P^D_i(t), Q^D_i(t)$ - Active / Reactive load demand for bus $i$
$\overline{\eta}^G_i , \underline{\eta}^G_i$ - Maximum / minimum ramp rate of generation $i$
$\overline{P}^\text{ESS}_{\text{c},i} , \overline{P}^\text{ESS}_{\text{d},i}$ - Maximum charging / discharging power of ESS $i$
$\overline{E}^\text{ESS}_{i} , \overline{E}^\text{ESS}_{i}$ - Upper / lower limit of energy stored in ESS $i$
$\eta^\text{ESS}_c, \eta^\text{ESS}_d$ - Charging / discharging efficiency of ESS 

$Ce(t)$ - Time-of-use price
$P^G_i(t)$ - Power output of the conventional generation $i$
$P^\text{RES}_i(t)$ - Power output of the RES $i$
$P^\text{ESS}_{\text{c},i} (t), P^\text{ESS}_{\text{d},i} (t)$ - Charging / discharging power of ESS $i$
$E^\text{ESS}_i(t)$ - Resource state of ESS $i$
$u_i(t)$ - Voltage amplitude of bus $i$
$\theta_i (t)$ - Voltage angle of bus $i$
$F(t)$ - Cost function of power system
$F_G(t)$ - Cost of conventional generation
$F_{\text{RES}}(t)$ - Operation cost Renewable energy curtailment, penalty term for abandoned energy of RES
$F_{\text{ESS}}(t)$ - Operation cost of ESS
$F_O(t)$ - Cost of purchasing electricity from and selling electricity to other power grids
$P_i(t) / Q_i(t)$ - Active / reactive power injection of bus $i$
$P_{ij}(t) / Q_{ij}(t)$ - Active / reactive branch power of line $ij$
$P_o(t)$ - Power output of connection lines with external grid
## Objective Function
$$
\text{min}\mathbb{E}(\sum^T_{t=1}F(t)) = \mathbb{E}(\sum^T_{t=1}F_G(t) + F_{\text{RES}}(t) + F_{\text{ESS}}(t) + F_O(t))
$$

$$
F_G(t) = \sum^{N^G}_{i=1}( a_i \Delta t + b_i P^G_i(t) \Delta t + c_i (P^G_i(t))^2 \Delta t)
$$

$$
F_\text{RES}(t) = \sum^{N^\text{RES}}_i \beta_\text{RES}(\overline{P}^\text{RES}_i(t) - P^\text{RES}_i(t)) \Delta t
$$
$$
F_\text{ESS}(t) = \sum^{N^\text{ESS}}_{i=1} \beta_\text{ESS}(P^\text{ESS}_{c,i}(t) + P^\text{ESS}_{d,j}(t)) \Delta t
$$
$$
F_O(t) = Ce(t)P_O(t)\Delta t
$$

### Power Balance Constraints

$$
P_i(t) = \sum_{a \in N^G_i} P^G_a(t) + \sum_{a \in N^\text{RES}_i} P^\text{RES}_a(t) - \sum_{b \in N^\text{ESS}_i}(P^\text{ESS}_{\text{c},b}(t) - P^\text{ESS}_{\text{d},b}) - P^D_i(t)
$$

$$
Q_i(t) = \sum_{a \in N^G_i} Q^G_a(t) + \sum_{a \in N^\text{RES}_i} Q^\text{RES}_a(t) - Q^D_i(t)
$$
$$
   P_i(t) = u_i(t) \sum_{j=1}^{N^\text{BUS}} u_j(t) (G_{ij} \cos(\theta_i(t) - \theta_j(t)) + B_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
$$
   Q_i(t) = u_i(t) \sum_{j=1}^{N^\text{BUS}} u_j(t) (G_{ij} \sin(\theta_i(t) - \theta_j(t)) - B_{ij} \cos(\theta_i(t) - \theta_j(t)))
$$
$$
\sum^{N^\text{BUS}}_{i=1} P_i(t) = \sum^{N^\text{G}}_{i=1} P^G_i(t) + \sum^{N^\text{RES}}_{i=1} P^\text{RES}_i(t) - \sum^{N^\text{ESS}}_{i=1}(P^\text{ESS}_{\text{c},i}(t) - P^\text{ESS}_{\text{d},i}) - \sum^{N^\text{BUS}}_{i=1} P^D_i(t)
$$

$$
\sum^{N^\text{BUS}}_{i=1} Q_i(t) = \sum^{N^\text{G}}_{i=1} Q^G_i(t) + \sum^{N^\text{RES}}_{i=1} Q^\text{RES}_i(t) - \sum^{N^\text{BUS}}_{i=1} Q^D_i(t)
$$


### Conventional Generation Constraints

$$
  \forall t, \underline{P}^G_i \leq P^G_i(t) \leq \overline{P}^G_i
$$
   $$
   \forall t, -\underline{\eta}^G_i \Delta t \leq P^G_i(t) - P^G_i(t-1) \leq \overline{\eta}^G_i \Delta t
   $$

### ESS Constraints
$$
   0 \leq P^\text{ESS}_{d,i}(t) \leq \overline{P}^\text{ESS}_{d,i}
$$
$$
   0 \leq P^\text{ESS}_{c,i}(t) \leq \overline{P}^\text{ESS}_{c,i}
$$
$$
   \underline{E}^\text{ESS}_i \leq E^\text{ESS}_i(t) \leq \overline{E}^\text{ESS}_i
$$
$$
   E^\text{ESS}_i(t) = E^\text{ESS}_i(t-1) + \left( \eta^\text{ESS}_\text{c} P^\text{ESS}_{\text{c},i}(t) - \frac{P^\text{ESS}_{\text{d},i}(t)}{\eta^\text{ESS}_{d}} \right) \Delta t
$$
$$
   P^\text{ESS}_{\text{c},i}(t) P^\text{ESS}_{\text{d},i}(t) = 0
$$




## RES power output constraints (?)
$$
   0 \leq P^\text{RES}_i(t) \leq \overline{P}^\text{RES}_i(t)
  $$



## Branch flow constraints (?)
   $$
   P_{ij}(t) = -u_i^2(t) G_{ij} + u_i(t) u_j(t) (G_{ij} \cos(\theta_i(t) - \theta_j(t)) + B_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
 $$
   Q_{ij}(t) = -u_i^2(t) B_{ij} + u_i(t) u_j(t) (B_{ij} \cos(\theta_i(t) - \theta_j(t)) + G_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
$$
P^2_{ij}(t) + Q^2_{ij}(t) \leq S^2_{ij}
$$

## Voltage constraints (?)
   $$
   \underline{u}_{i} \leq u_i(t) \leq \overline{u}_{i}
$$


## A hybrid EP and SQP for dynamic economic dispatch with nonsmooth fuel cost function


**Objective Function**
$$
minF = \sum^T_{t=1}{\sum^n_{i=1} F_{it}(P_{it})}
$$
$F$ - total operating cost over the whole dispatch periods
$T$ - number of hours in the time horizon
$n$ - number of dispatchable units
$F_{it}(P_{it})$ - individual generation production cost in terms of real power output $P_i$ at time $t$

**Real Power Balance Constraint*

$$
\sum^n_{i=1}P_{ij} - P_{Dj} = 0
$$

$P_{Dj}$ - total assumed load demand during $j$-th interval

**Real Power Operating Limits**:

$$
P_{i,\text{min}} \leq P_i \leq P_{i,\text{max}}
$$

$P_{i,\text{min}}$ , $P_{i,\text{max}}$ - minimum and maximum real power outputs of the $i$-th generator

**Generating Unit Ramp Rate**

$$
P_{it} - P_{i(t-1)} \leq \text{UR}_i 
$$
$$
P_{i(t-1)} - P_{it} \leq \text{DR}_i 
$$
$\text{UR}_i$ , $\text{DR}_i$ - ramp-up and ramp-down rate limits of *i*-th generator


## A scalable graph reinforcement learning algorithm based stochastic dynamic dispatch of power system under high penetration of renewable energy

**Objective Function**:
$$
\text{min}\mathbb{E}(\sum^T_{t=1}F(t)) = \mathbb{E}(\sum^T_{t=1}F_G(t) + F_{\text{RES}}(t) + F_{\text{ESS}}(t))
$$




$$
F_G(t) = \sum^{N^G}_{i=1}( a_i \Delta t + b_i P^G_i(t) \Delta t + c_i (P^G_i(t))^2 \Delta t)
$$

$$
F_\text{RES}(t) = \sum^{N^\text{RES}}_i \beta_\text{RES}(\overline{P}^\text{RES}_i(t) - P^\text{RES}_i(t)) \Delta t
$$
$$
F_\text{ESS}(t) = \sum^{N^\text{ESS}}_{i=1} \beta_\text{ESS}(P^\text{ESS}_{c,i}(t) + P^\text{ESS}_{d,j}(t)) \Delta t
$$

**Power Balance Constraints**:

$$
P_i(t) = \sum_{a \in N^G_i} P^G_a(t) + \sum_{a \in N^\text{RES}_i} P^\text{RES}_a(t) - \sum_{b \in N^\text{ESS}_i}(P^\text{ESS}_{\text{c},b}(t) - P^\text{ESS}_{\text{d},b}) - P^D_i(t)
$$

$$
Q_i(t) = \sum_{a \in N^G_i} Q^G_a(t) + \sum_{a \in N^\text{RES}_i} Q^\text{RES}_a(t) - Q^D_i(t)
$$
$$
   P_i(t) = u_i(t) \sum_{j=1}^{N^\text{BUS}} u_j(t) (G_{ij} \cos(\theta_i(t) - \theta_j(t)) + B_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
$$
   Q_i(t) = u_i(t) \sum_{j=1}^{N^\text{BUS}} u_j(t) (G_{ij} \sin(\theta_i(t) - \theta_j(t)) - B_{ij} \cos(\theta_i(t) - \theta_j(t)))
$$

**Branch flow constraints**:

   $$
   P_{ij}(t) = -u_i^2(t) G_{ij} + u_i(t) u_j(t) (G_{ij} \cos(\theta_i(t) - \theta_j(t)) + B_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
 $$
   Q_{ij}(t) = -u_i^2(t) B_{ij} + u_i(t) u_j(t) (B_{ij} \cos(\theta_i(t) - \theta_j(t)) + G_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
$$
P^2_{ij}(t) + Q^2_{ij}(t) \leq S^2_{ij}
$$

**Voltage constraints**:
   $$
   \underline{u}_{i} \leq u_i(t) \leq \overline{u}_{i}
$$

**Conventional thermal generation constraints**:
$$
   \underline{P}^G_i \leq P^G_i(t) \leq \overline{P}^G_i
$$
   $$
   -\underline{\eta}^G_i \Delta t \leq P^G_i(t) - P^G_i(t-1) \leq \overline{\eta}^G_i \Delta t
   $$ 
**ESS constraints**:
$$
   0 \leq P^\text{ESS}_{d,i}(t) \leq \overline{P}^\text{ESS}_{d,i}
$$
$$
   0 \leq P^\text{ESS}_{c,i}(t) \leq \overline{P}^\text{ESS}_{c,i}
$$
$$
   \underline{E}^\text{ESS}_i \leq E^\text{ESS}_i(t) \leq \overline{E}^\text{ESS}_i
$$
$$
   E^\text{ESS}_i(t) = E^\text{ESS}_i(t-1) + \left( \eta^\text{ESS}_\text{c} P^\text{ESS}_{\text{c},i}(t) - \frac{P^\text{ESS}_{\text{d},i}(t)}{\eta^\text{ESS}_{d}} \right) \Delta t
$$
$$
   P^\text{ESS}_{\text{c},i}(t) P^\text{ESS}_{\text{d},i}(t) = 0
$$

**RES power output constraints**:
$$
   0 \leq P^\text{RES}_i(t) \leq \overline{P}^\text{RES}_i(t)
  $$

## A Novel Graph Reinforcement Learning Approach for Stochastic Dynamic Economic Dispatch under High Penetration of Renewable Energy

**Objective Function**:
$$
\text{min}\mathbb{E}(\sum^T_{t=1}F(t)) = \mathbb{E}(\sum^T_{t=1}F_G(t) + F_{\text{RES}}(t) + F_{\text{ESS}}(t) + F_O(t))
$$

$$
F_G(t) = \sum^{N^G}_{i=1}( a_i \Delta t + b_i P^G_i(t) \Delta t + c_i (P^G_i(t))^2 \Delta t)
$$

$$
F_\text{RES}(t) = \sum^{N^\text{RES}}_i \beta_\text{RES}(\overline{P}^\text{RES}_i(t) - P^\text{RES}_i(t)) \Delta t
$$
$$
F_\text{ESS}(t) = \sum^{N^\text{ESS}}_{i=1} \beta_\text{ESS}(P^\text{ESS}_{c,i}(t) + P^\text{ESS}_{d,j}(t)) \Delta t
$$
$$
F_O(t) = Ce(t)P_O(t)\Delta t
$$

**Power Balance Constraints**:
$$
\sum^{N^\text{BUS}}_{i=1} P_i(t) = \sum^{N^\text{G}}_{i=1} P^G_i(t) + \sum^{N^\text{RES}}_{i=1} P^\text{RES}_i(t) - \sum^{N^\text{ESS}}_{i=1}(P^\text{ESS}_{\text{c},i}(t) - P^\text{ESS}_{\text{d},i}) - \sum^{N^\text{BUS}}_{i=1} P^D_i(t)
$$

$$
\sum^{N^\text{BUS}}_{i=1} Q_i(t) = \sum^{N^\text{G}}_{i=1} Q^G_i(t) + \sum^{N^\text{RES}}_{i=1} Q^\text{RES}_i(t) - \sum^{N^\text{BUS}}_{i=1} Q^D_i(t)
$$


**Conventional generation constraints**:
$$
   \underline{P}^G_i \leq P^G_i(t) \leq \overline{P}^G_i
$$
   $$
   -\underline{\eta}^G_i \Delta t \leq P^G_i(t) - P^G_i(t-1) \leq \overline{\eta}^G_i \Delta t
   $$ 
**ESS constraints**:
$$
   0 \leq P^\text{ESS}_{d,i}(t) \leq \overline{P}^\text{ESS}_{d,i}
$$
$$
   0 \leq P^\text{ESS}_{c,i}(t) \leq \overline{P}^\text{ESS}_{c,i}
$$
$$
   \underline{E}^\text{ESS}_i \leq E^\text{ESS}_i(t) \leq \overline{E}^\text{ESS}_i
$$
$$
   E^\text{ESS}_i(t) = E^\text{ESS}_i(t-1) + \left( \eta^\text{ESS}_\text{c} P^\text{ESS}_{\text{c},i}(t) - \frac{P^\text{ESS}_{\text{d},i}(t)}{\eta^\text{ESS}_{d}} \right) \Delta t
$$
$$
   P^\text{ESS}_{\text{c},i}(t) P^\text{ESS}_{\text{d},i}(t) = 0
$$



## Dynamic energy dispatch strategy for integrated energy system based on improved deep reinforcement learning

**Objective Function**:
$$
F = \text{min}\sum^T_{t=1} (F_O(t) +  F_{\text{ESS}}(t))
$$

$\varepsilon_p (t)$  - purchasing electricity price
$\varepsilon_s (t)$ - selling electricity price
$\varepsilon_\text{gas} (t)$  - natural gas price
$P_\text{CHP} (t)$ - power output of combined heat and power
$\eta_\text{CHP}$

$$
F_O(t) = \left( P_O(t)\frac{\varepsilon_p(t) + \varepsilon_s(t)}{2} + |P_O(t)|\frac{\varepsilon_p(t) + \varepsilon_s(t)}{2}  + \varepsilon_\text{gas}(t) \left( \frac{P_\text{CHP}(t)}{\eta_\text{CHP}} + \frac{h_{GB} (t)}{\eta_\text{GB}} \right)  \right) \Delta t
$$
$$
F_\text{ESS}(t) = \sum^{N^\text{ESS}}_{i=1} \beta_\text{ESS}(P^\text{ESS}_{c,i}(t) + P^\text{ESS}_{d,j}(t)) \Delta t
$$

**Power Balance Constraints**:

$$
P_i(t) = \sum_{a \in N^G_i} P^G_a(t) + \sum_{a \in N^\text{RES}_i} P^\text{RES}_a(t) - \sum_{b \in N^\text{ESS}_i}(P^\text{ESS}_{\text{c},b}(t) - P^\text{ESS}_{\text{d},b}) - P^D_i(t)
$$

$$
Q_i(t) = \sum_{a \in N^G_i} Q^G_a(t) + \sum_{a \in N^\text{RES}_i} Q^\text{RES}_a(t) - Q^D_i(t)
$$
$$
   P_i(t) = u_i(t) \sum_{j=1}^{N^\text{BUS}} u_j(t) (G_{ij} \cos(\theta_i(t) - \theta_j(t)) + B_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
$$
   Q_i(t) = u_i(t) \sum_{j=1}^{N^\text{BUS}} u_j(t) (G_{ij} \sin(\theta_i(t) - \theta_j(t)) - B_{ij} \cos(\theta_i(t) - \theta_j(t)))
$$

**Branch flow constraints**:

   $$
   P_{ij}(t) = -u_i^2(t) G_{ij} + u_i(t) u_j(t) (G_{ij} \cos(\theta_i(t) - \theta_j(t)) + B_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
 $$
   Q_{ij}(t) = -u_i^2(t) B_{ij} + u_i(t) u_j(t) (B_{ij} \cos(\theta_i(t) - \theta_j(t)) + G_{ij} \sin(\theta_i(t) - \theta_j(t)))
$$
$$
P^2_{ij}(t) + Q^2_{ij}(t) \leq S^2_{ij}
$$

**Voltage constraints**:
   $$
   \underline{u}_{i} \leq u_i(t) \leq \overline{u}_{i}
$$

**Conventional thermal generation constraints**:
$$
   \underline{P}^G_i \leq P^G_i(t) \leq \overline{P}^G_i
$$
   $$
   -\underline{\eta}^G_i \Delta t \leq P^G_i(t) - P^G_i(t-1) \leq \overline{\eta}^G_i \Delta t
   $$ 
**ESS constraints**:
$$
   0 \leq P^\text{ESS}_{d,i}(t) \leq \overline{P}^\text{ESS}_{d,i}
$$
$$
   0 \leq P^\text{ESS}_{c,i}(t) \leq \overline{P}^\text{ESS}_{c,i}
$$
$$
   \underline{E}^\text{ESS}_i \leq E^\text{ESS}_i(t) \leq \overline{E}^\text{ESS}_i
$$
$$
   E^\text{ESS}_i(t) = E^\text{ESS}_i(t-1) + \left( \eta^\text{ESS}_\text{c} P^\text{ESS}_{\text{c},i}(t) - \frac{P^\text{ESS}_{\text{d},i}(t)}{\eta^\text{ESS}_{d}} \right) \Delta t
$$
$$
   P^\text{ESS}_{\text{c},i}(t) P^\text{ESS}_{\text{d},i}(t) = 0
$$

**RES power output constraints**:
$$
   0 \leq P^\text{RES}_i(t) \leq \overline{P}^\text{RES}_i(t)
  $$
