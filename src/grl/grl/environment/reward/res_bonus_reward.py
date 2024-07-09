import grid2op
import numpy as np
from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.baseReward import BaseReward


class RESBonusReward(BaseReward):
    """
    Economic Reward class for the grid2op environment with bonus for used renewable energy.

    The reward is computed as follows:
    - -1 is there is an error in the environment
    - Zero if the action is illegal or ambiguous
    - Else is the sum of the cost saved and the bonus for used renewable energy.

    """

    def __init__(self, res_bonus: float, logger=None):
        """
        Initialize the RESBonusReward class.

        :param res_bonus: renewable energy bonus
        :param logger: logger
        """
        if res_bonus is None or res_bonus < 0.0 or res_bonus > 1.0:
            raise ValueError("res_bonus must be greater than 0")
        BaseReward.__init__(self, logger=logger)
        self.reward_min = np.float32(-1.0)
        self.reward_max = np.float32(1.0)
        self.res_bonus = res_bonus
        self.worst_cost = 0.0
        self.n_renewables = 0

    def initialize(self, env: grid2op.Environment):
        """
        Initialize the reward.

        :param env: grid2op environment
        """

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

        avg_cost_per_mw = np.float32(np.sum(env.gen_cost_per_MW) / (env.n_gen - n_res))

        self.worst_cost = np.float32(
            (env.gen_cost_per_MW * env.gen_pmax).sum() * env.delta_time_seconds / 3600.0
        )
        self.res_bonus = np.float32(self.res_bonus * avg_cost_per_mw)

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

        # compute the renewable energy used and the max renewable energy before curtailment
        res = 0
        res_max = 0
        for i in range(0, env.n_gen):
            if env.gen_renewable[i]:
                res += np.float32(env.get_obs(_do_copy=False).gen_p[i])
                res_max += env.get_obs(_do_copy=False).gen_p_before_curtail[i]

        res_term = self.res_bonus * res * env.delta_time_seconds / 3600.0

        # compute the cost of the grid
        r = cost_saved + res_term

        # calculate the variable higher bound of the reward
        high = np.float32(
            self.worst_cost
            + self.res_bonus * np.sum(res_max) * env.delta_time_seconds / 3600.0
        )

        return np.interp(r, [0, high], [0, self.reward_max])
