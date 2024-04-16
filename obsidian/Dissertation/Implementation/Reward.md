
$\mathbb{E}$ - expectation operator
$T$ - Set of dispatch stages
$N^\text{BUS}$ - Set of buses
$N^G, N^\text{RES}, N^\text{ESS}$ - Set of conventional generation, RES and ESS
$N^G_i, N^\text{RES}_i, N^\text{ESS}_i$ - Set of conventional generation, RES and ESS connected to node i
$\Delta t$ - time interval
$a_i, b_i, c_i$ - Cost coefficient of conventional generation $i$
$\beta_\text{RES}$ - Penalty coefficient for RES Curtailment
$\beta_\text{ESS}$ - Operation cost coefficient for ESS Curtailment
$\overline{P}^\text{RES}_i(t)$  - Maximum active power output of RES $i$
$\overline{P}^G_i , \underline{P}^G_i$ - Maximum / minimum power output of conventional generation $i$
$\overline{u}_i, \underline{u}_i$ - Maximum / minimum voltage magnitude of bus $i$
$C_i$ - Cost per MW of generation $i$
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

**Worst Cost for Non-renewable Generators**
$$
F_\text{worst} = \sum_{i=0}^{N^G} \overline{P^G_i} C_i 
$$
**Cost for Non-renewable Generators**
$$ 
F_{G}(t) = \sum_{i=0}^{N^G} P^G_i (t) C_i 
$$
**Cost Difference for Non-renewable Generators (More is better)**:
$$
F_\text{G,diff}(t) = \sum_{i=0}^{N^G} \overline{P^G_i} C_i - P^G_i(t) C_i 
$$
**Cost Ratio for Non-Renewable Generators (More is worse):**
$$
F_\text{G,ratio}(t) = \sum_{i=0}^{N^G} \frac{P^G_i(t) C_i}{\overline{P^G_i} C_i} \times 100
$$
**Cost Difference for Renewable Generators (Less is better)**:
$$
F_\text{RES,diff}(t) = \sum^{N^\text{RES}}_{i = 0} \overline{P^\text{RES}_i}(t) - P^\text{RES}_i (t)
$$
**Cost Ratio for Renewable Generators (More is better)**:

$$
F_\text{RES,ratio}(t) = \sum^{N^\text{RES}}_{i = 0} \frac{P^\text{RES}_i(t)}{\overline{P^\text{RES}_i}(t)} \times 100
$$

$$
F(t) = F^G(t) + \beta F^\text{RES}(t)+F^\text{ESS}(t)
$$
