```python
import grid2op
from grid2op.Action import DontAct
from grid2op.Opponent import BaseOpponent, NeverAttackBudget
env_name = ...

# if you want to disable the opponent you can do (grid2op >= 1.9.4)
kwargs_no_opp = grid2op.Opponent.get_kwargs_no_opponent()
env_no_opp = grid2op.make(env_name, **kwargs_no_opp)
# and there the opponent is disabled

# or, in a more complex fashion (or for older grid2op version <= 1.9.3)
env_without_opponent = grid2op.make(env_name,
                                    opponent_attack_cooldown=999999,
                                    opponent_attack_duration=0,
                                    opponent_budget_per_ts=0,
                                    opponent_init_budget=0,
                                    opponent_action_class=DontAct,
                                    opponent_class=BaseOpponent,
                                    opponent_budget_class=NeverAttackBudget,
                                    ...  # other arguments pass to the "make" function
                                    )
```
