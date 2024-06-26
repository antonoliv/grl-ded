import grid2op.Action.baseAction
import numpy as np
from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.baseReward import BaseReward


class RESPenaltyReward(BaseReward):
    """
    Economic Reward class for the grid2op environment with penalty for wasted renewable energy.

    The reward is computed as follows:
    - -1 is there is an error in the environment
    - Zero if the action is illegal or ambiguous
    - Else is the difference between the cost saved and the penalty for wasted renewable energy.

    """

    def __init__(self, res_penalty: float, logger=None):
        """
        Initialize the RESPenaltyReward class.

        :param res_penalty: renewable energy penalty
        :param logger: logger
        """
        if res_penalty is None or res_penalty < 0.0 or res_penalty > 1.0:
            raise ValueError("res_penalty must be between 0 and 1")

        BaseReward.__init__(self, logger=logger)
        self.reward_min = np.float32(-1.0)
        self.reward_max = np.float32(1.0)
        self.res_penalty = res_penalty
        self.worst_cost = 0.0
        self.avg_cost_per_mw = 0.0
        self.n_renewables = 0

    def initialize(self, env: grid2op.Environment):
        """
        Initialize the reward.

        :param env: grid2op environment
        """

        # Check if the environment has generators
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException(
                "Impossible to use the EconomicReward reward with an environment without generators"
                "cost. Please make sure env.redispatching_unit_commitment_availble is available."
            )

        n_res = 0

        res_gmax = 0
        for i in range(0, env.n_gen):
            if env.gen_renewable[i]:
                n_res += 1
                res_gmax += env.gen_pmax[i]

        # Average cost per MW
        self.avg_cost_per_mw = np.float32(
            np.sum(env.gen_cost_per_MW) / (env.n_gen - n_res)
        )

        # Worst cost of the grid
        self.worst_cost = np.float32(
            (env.gen_cost_per_MW * env.gen_pmax).sum() * env.delta_time_seconds / 3600.0
        )

        # Number of non-renewable generators
        self.n_renewables = n_res

    def __call__(
        self,
        action: grid2op.Action,
        env: grid2op.Environment,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ):
        """
        Compute the reward for the given action.

        :param action:              grid2op action
        :param env:                 grid2op environment
        :param has_error:           true if there is an error in the environment
        :param is_done:             true if the episode is done
        :param is_illegal:          true if the action is illegal
        :param is_ambiguous:        true if the action is ambiguous
        :return:
        """

        if has_error:
            # if there is an error in the environment reward_min
            return self.reward_min

        if is_illegal or is_ambiguous:
            # if the action is illegal or ambiguous return 0
            return 0.0

        # compute the cost
        cost = np.float32(
            (env.get_obs(_do_copy=False).gen_p * env.gen_cost_per_MW).sum()
            * env.delta_time_seconds
            / 3600.0
        )

        # compute the cost saved
        cost_saved = self.worst_cost - cost

        # compute the renewable energy wasted and the max renewable energy before curtailment
        res = 0
        res_max = 0
        for i in range(0, env.n_gen):
            if env.gen_renewable[i]:
                res += np.float32(
                    env.get_obs(_do_copy=False).gen_p_before_curtail[i]
                    - env.get_obs(_do_copy=False).gen_p[i]
                )
                res_max += np.float32(
                    env.get_obs(_do_copy=False).gen_p_before_curtail[i]
                )

        # res
        res_term = (
            self.res_penalty
            * self.avg_cost_per_mw
            * res
            * env.delta_time_seconds
            / 3600.0
        )

        # compute reward
        r = cost_saved - res_term

        # calculate the variable lower bound of the reward
        low = np.float32(
            self.res_penalty * np.sum(res_max) * env.delta_time_seconds / 3600.0 * -1
        )

        # scale the reward between 0 and reward_max
        return np.interp(r, [low, self.worst_cost], [0, self.reward_max])
