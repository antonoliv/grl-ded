### Generators

Elements connected to the power grid with the main purpose of producing power.
#### Static Properties
|Name|Type|Description|
|---|---|---|
|n_gen|int|Total number of generators on the grid|
|name_gen|vect, string|Names of all the generators|
|gen_to_subid|vect, int|To which substation each generator is connected|
|gen_to_sub_pos|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|
|gen_pos_topo_vect|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|
|* gen_type|vect, string|Type of generator, among “nuclear”, “hydro”, “solar”, “wind” or “thermal”|
|* gen_renewable|vect, bool|Is the generator “renewable”|
|* gen_pmin|vect, float|Minimum production physically possible for each generator, in MW|
|* gen_pmax|vect, float|Maximum production physically possible for each generator, in MW|
|* gen_redispatchable|vect, bool|For each generator, indicates if it can be “dispatched” see the subsection about the action for more information on dispatch|
|* gen_max_ramp_up|vect, float|For each generator, indicates the maximum values the power can vary (upward) between two consecutive steps in MW. See the subsection about the equations for more information|
|* gen_max_ramp_down|vect, float|For each generator, indicates the maximum values the power can vary (downward) between two consecutive steps in MW. See the subsection about the equations for more information|
|* gen_min_uptime|vect, int|(currently unused) For each generator, indicates the minimum time a generator need to be “on” before being turned off.|
|* gen_min_downtime|vect, int|(currently unused) For each generator, indicates the minimum time a generator need to be “off” before being turned on again.|
|* gen_cost_per_MW|vect, float|(will change in the near future) Cost of production, in $ / MWh (in theory) but in $ / (MW . step) (each step “costs” prod_p * gen_cost_per_MW)|
|* gen_startup_cost|vect, float|(currently unused) Cost to turn on each generator (in $)|
|* gen_shutdown_cost|vect, float|(currently unused) Cost to turn off each generator (in $)|
\* - optional attribute depending on scenario

#### Modifiable Attributes

| Name | Type | Description |
| ---- | ---- | ---- |
| gen_set_bus | vect, int | set the bus to which the generator is connected |
| gen_change_bus | vect, bool | change the bus to which the generator is connected |
| redispatch | vect, float | will apply some redispatching a generator |
| (internal) prod_p  | vect, float | change active production |
| (internal) prod_v | vect, float | change the voltage setpoint |
| curtail | vect, float | will apply some curtailment on a generator |

#### Observable Attributes

| Name | Type | Description |
| ---- | ---- | ---- |
| gen_p | vect, float | the current active production of each generators, in MW. |
| gen_q | vect, float | the current reactive production of each generators, in MVAr. |
| gen_v | vect, float | the voltage of the bus at which the generator is connected, in kV. |
| target_dispatch | vect, float | the bus to which each generators is connected. |
| actual_dispatch | vect, float | the target values given by the agent to the environment, in MW. |
| gen_bus | vect, int | actual dispatch: the values the environment was able to provide as redispatching, in MW. |
| curtailment | vect, float | give the ratio of curtailment for each generator. **NB** it will always be 0.0 for non renewable generator. **NB** the property curtailment_mw also exists if you want to convert the curtailment, normally expressed in ratio of gen_pmax as a curtailment in MW. |
| gen_p_before_curtail | vect, float | give the amount of production for each renewable generators if no curtailment were applied. **NB** by convention it will be 0.0 for non renewable generator |
| curtailment_limit | vect, float | add the limits of all the past curtailment actions. |
|  |  |  |



### Loads

An element that consumes power from the power grid.
#### Static Properties
|Name|Type|Description|
|---|---|---|
|n_load|int|Total number of loads on the grid|
|name_load|vect, string|Names of all the loads|
|load_to_subid|vect, int|To which substation each load is connected|
|load_to_sub_pos|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|
|load_pos_topo_vect|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|

#### Modifiable Attributes
| Name | Type | Description |
| ---- | ---- | ---- |
| load_set_bus | vect, int | set the bus to which the load is connected. |
| load_change_bus | vect, bool | change the bus to which the load is connected. |
| load_p | vect, float | change the active consumption of a load. |
| load_q | vect, float | change the reactive consumption of a load. |

#### Observable Attributes
| Name  | Type  | Description  |
|---|---|---|
|load_p|vect, float|the current active consumption of each load, in MW. |
|load_q|vect, float|the current reactive consumption of each load, in MVAr. |
|load_v|vect, float|the voltage of the bus at which the load is connected, in kV. |
|load_bus|vect, int|the bus to which each load is connected. |


### Powerlines

Elements of the power grid that allow power to flow from one point to another
#### Static Properties
|Name|Type|Description|
|---|---|---|
|n_line|int|Total number of lines on the grid|
|name_line|vect, string|Names of all the lines|
|line_or_to_subid|vect, int|To which substation each line (origin side) is connected|
|line_ex_to_subid|vect, int|To which substation each line (extremity side) is connected|
|line_or_to_sub_pos|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|
|line_ex_to_sub_pos|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|
|line_or_pos_topo_vect|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|
|line_ex_pos_topo_vect|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|

#### Modifiable Attributes
| Name | Type | Description |
| ---- | ---- | ---- |
| line_or_set_bus | vect, int | set the bus to which the origin side of the powerline is connected. |
| line_ex_set_bus | vect, int | set the bus to which the extremity side of the powerline is connected. |
| line_ex_change_bus | vect, int | change the bus to which the origin side of a powerline is connected. |
| line_or_change_bus | vect, int | change the bus to which the extremity side of a powerline is connected. |
| line_set_status | vect, int | set the status (connected / disconnected) of a powerline. |
| line_change_status | vect, int | change the status of a powerline. |

#### Observable Attributes
| Name | Type | Description |
| ---- | ---- | ---- |
| a_or | vect, float | intensity flows (also known as current flows) at the “origin side” of the powerlines, measured in Amps (A). |
| a_ex | vect, float | intensity flows (also known as current flows) at the “extremity side” of the powerlines, measured in Amps (A). |
| p_or | vect, float | active flows at the “origin side” of the powerlines, measured in Mega Watt (MW). |
| p_ex | vect, float | active flows at the “extremity side” of the powerlines, measured in Mega Watt (MW). |
| q_or | vect, float | reactive flows at the “origin side” of the powerlines, measured in Mega Volt Amps reactive (MVAr). |
| q_ex | vect, float | reactive flows at the “extremity side” of the powerlines, measured in Mega Volt Amps reactive (MVAr). |
| v_or | vect, float | voltage magnitude at the bus to which the “origin side” of the powerline is connected, measured in kilo Volt (kV). |
| v_ex | vect, float | voltage magnitude at the bus to which the “extremity side” of the powerline is connected, measured in kilo Volt (kV). |
| rho | vect, float | relative flows on each powerlines. It is the ratio of the flow on the powerline divided by its thermal limit. |
| line_status | vect, bool | gives the status (connected / disconnected) of each powerlines. |
| timestep_overflow | vect, int | for each powerline, returns the number of steps since this powerline is on overflow. This is given in number of steps (no units). Most of the time it will be 0 meaning the powerline is not on overflow. |
| time_before_cooldown_line | vect, int | number of steps you need to wait before being able to change the status of powerline again. |
| time_next_maintenance | vect, int | indicates the next scheduled maintenance operation on each of the powerline. |
| duration_next_maintenance | vect, int | indicates the duration of the next scheduled maintenance for each powerline. |
| thermal_limit | vect, float | for each powerline, it gives its “thermal limit” |
| line_or_bus | vect, int | for each powerline, it gives the busbars (usually -1, 1 or 2) at which the “origin side” of the powerline is connected. |
| line_ex_bus | vect, int | for each powerline, it gives the busbars (usually -1, 1 or 2) at which the “extremity side” of the powerline is connected. |


### Storage Units
#### Static Properties
|Name|Type|Description|
|---|---|---|
|n_storage|int|Number of storage units on the grid|
|name_storage|vect, str|Name of each storage units|
|storage_to_subid|vect, int|Id of the substation to which each storage units is connected|
|storage_to_sub_pos|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|
|storage_pos_topo_vect|vect, int|Internal, see [Creating a new backend](https://grid2op.readthedocs.io/en/latest/createbackend.html#create-backend-module)|
|storage_type|vect, str|Type of storage, among “battery” or “pumped_storage”|
|storage_Emax|vect, float|For each storage unit, the maximum energy it can contains, in MWh|
|storage_Emin|vect, float|For each storage unit, the minimum energy it can contains, in MWh|
|storage_max_p_prod|vect, float|For each storage unit, the maximum power it can give to the grid, in MW|
|storage_max_p_absorb|vect, float|For each storage unit, the maximum power it can take from the grid, in MW|
|storage_marginal_cost|vect, float|For each storage unit, the cost for taking / adding 1 MW to the grid, in $|
|storage_loss|vect, float|For each storage unit, the self discharge, in MW, of the unit|
|storage_charging_efficiency|vect, float|For each storage unit, the “charging efficiency” (see bellow)|
|storage_discharging_efficiency|vect, float|For each storage unit, the “discharging efficiency” (see bellow)|

#### Modifiable Attributes
| Name | Type | Description |
| ---- | ---- | ---- |
| storage_set_bus | vect, int | set the bus to which the storage unit is connected. |
| storage_change_bus | vect, int | change the bus to which the storage unit is connected. |
| storage_p | vect, float | will tell the storage unit you want to get a given amount of power on the grid. |

#### Observable Attributes
| Name | Type | Description |
| ---- | ---- | ---- |
| storage_power | vect, float | the power that is actually produced / absorbed by every storage unit. |
| storage_power_target | vect, float | the power that was required from the last action of the agent, in MW |
| storage_charge | vect, float | the state of charge of each storage unit, in MWh. |
| storage_bus | vect, float | for each storage unit, it gives the busbars (usually -1, 1 or 2) at which it is connected. |


### Substations
#### Static Properties
| Name | Type | Description |
| ---- | ---- | ---- |
| n_sub | int | Total number of substation on the grid |
| sub_info | vect, int | For each substations, gives the number of elements (side of powerline, load, generator or storage unit) connected to it. |
| dim_topo | int | Total number of elements (side of powerline, load, generator or storage unit) on the grid |
| name_sub | vect, str | Name of each substation |

#### Modifiable Attributes
| Name | Type | Description |
| ---- | ---- | ---- |
| set_bus | vect, int | perform an action of type set_bus on a given set of elements. |
| change_bus | vect, int | perform an action of type change_bus on a given set of elements. |

#### Observable Attributes
| Name | Type | Description |
| ---- | ---- | ---- |
| topo_vect | vect, int | for each element of the grid, gives on which busbar this elemement is connected. |
| time_before_cooldown_sub | vect, int | number of steps you need to wait before being able to change the topology of each substation again. |


### Shunts (optional)

A low-resistance electrical path that allows current to flow around another component or part of a circuit. Shunts are often used for measuring current in an ammeter or for providing a reference voltage in a voltmeter.

#### Static Properties

#### Modifiable Attributes

#### Observable Attributes