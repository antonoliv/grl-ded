import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float


class DynamicEconomicReward(BaseReward):
    """
    This reward computes the marginal cost of the powergrid. As RL is about maximising a reward, while we want to
    minimize the cost, this class also ensures that:

    - the reward is positive if there is no game over, no error etc.
    - the reward is inversely proportional to the cost of the grid (the higher the reward, the lower the economic cost).

    Examples
    ---------
    You can use this reward in any environment with:

    .. code-block:: python

        import grid2op
        from grid2op.Reward import EconomicReward

        # then you create your environment with it:
        NAME_OF_THE_ENVIRONMENT = "l2rpn_case14_sandbox"
        env = grid2op.make(NAME_OF_THE_ENVIRONMENT,reward_class=EconomicReward)
        # and do a step with a "do nothing" action
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_space())
        # the reward is computed with the EconomicReward class

    """

    def __init__(self, res_penalty: float, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)
        self.res_penalty = res_penalty
        self.worst_cost = None

    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException(
                "Impossible to use the EconomicReward reward with an environment without generators"
                "cost. Please make sure env.redispatching_unit_commitment_availble is available."
            )
        self.worst_cost = dt_float((env.gen_cost_per_MW * env.gen_pmax).sum() * env.delta_time_seconds / 3600.0)

        n_res = 0
        res_gmax = 0
        for i in range(0, env.n_gen):
                if env.gen_renewable[i]:
                    n_res += 1
                    res_gmax += env.gen_pmax[i]

        self.n_renewables = n_res
        self.res_gmax = res_gmax
        

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            r = self.res_penalty * self.res_gmax * -1
        else:
            cost = dt_float((env.get_obs(_do_copy=False).gen_p * env.gen_cost_per_MW).sum() * env.delta_time_seconds / 3600.0)
            cost_saved = self.worst_cost - cost

            res = 0
            for i in range(0, env.n_gen):
                if env.gen_renewable[i]:
                    res += dt_float(env.get_obs(_do_copy=False).gen_p_before_curtail[i] - env.get_obs(_do_copy=False).gen_p[i])

            res_term = self.res_penalty * res


            # compute the cost of the grid
            r = cost_saved - res_term
            

        r = np.interp(
            r, [dt_float(0.0) - self.res_penalty * self.res_gmax, self.worst_cost], [self.reward_min, self.reward_max]
        )
        return dt_float(r)
