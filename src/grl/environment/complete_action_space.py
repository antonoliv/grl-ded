import grid2op
import numpy as np
from gymnasium import spaces


class CompleteActionSpace(spaces.Box):
    """
    Custom Action Space for the environment.

    Main functionalities:
    - Handle redispatch and curtailment actions
    - Handle scaled ([-1, 1]) and unscaled actions
    - Reconnect lines after x timesteps
    - Limit the curtailment action
    """

    def __init__(
        self,
        init_env: grid2op.Environment,
        scaled: bool,
        curtail_limit: float,
    ):
        """
        Init function for the CompleteActionSpace class.

        :param init_env:        grid2op environment
        :param scaled:          true if actions are scaled
        :param curtail_limit:   limit for the curtailment action
        """
        if init_env is None:
            raise ValueError("Environment is not initialized")
        if scaled is None:
            raise ValueError("Scaled is not initialized")
        if curtail_limit is None or curtail_limit < 0.0 or curtail_limit > 1.0:
            raise ValueError("Curtail limit should be between 0 and 1")

        self.curtail_limit = 0.0
        self.reconnect_delta = 10  # reconnect line after x timesteps

        # Initialize parameters
        self._init_env = init_env  # initial grid2op environment
        self.scaled = scaled  # true if actions are scaled
        self.update_curtail_limit(curtail_limit)  # limit for the curtailment action
        self.n_gen = init_env.n_gen  # number of generators

        self.gen_max = np.ones(
            self.n_gen, dtype=np.float32
        )  # maximum action for each generator
        self.gen_min = np.zeros(
            self.n_gen, dtype=np.float32
        )  # minimum action for each generator

        self.gen_max[self._init_env.gen_redispatchable] = (
            self._init_env.gen_max_ramp_up[self._init_env.gen_redispatchable]
        )
        self.gen_min[self._init_env.gen_redispatchable] = (
            -self._init_env.gen_max_ramp_down[self._init_env.gen_redispatchable]
        )

        # Initialize the disconnected time for each line
        self.disconnected_t = np.zeros(self._init_env.n_line, dtype=np.int32)

        if self.scaled:
            # if actions are scaled, the action space is [-1, 1]
            spaces.Box.__init__(
                self, shape=(self.n_gen,), low=-1, high=1, dtype=np.float32
            )
        else:
            # if actions are not scaled, the action space is [gen_min, gen_max]
            spaces.Box.__init__(
                self,
                shape=(self.n_gen,),
                low=self.gen_min,
                high=self.gen_max,
                dtype=np.float32,
            )

    def from_gym(self, action: np.ndarray) -> grid2op.Action:
        """
        Convert gym action into grid2op action.

        :param action: gymnasium action
        :return: grid2op action
        """

        # Initialize the grid2op action
        set_line = []

        # Get the current observation
        obs = self._init_env.current_obs

        # Line status
        for i in range(0, self._init_env.n_line):

            if obs.line_status[i] == 0:
                # if line is disconnected, increment the disconnected time
                self.disconnected_t[i] += 1
            elif self.disconnected_t[i] > 0:
                # if line is connected, reset the disconnected time
                self.disconnected_t[i] = 0

            if self.disconnected_t[i] > self.reconnect_delta:
                # reconnect line after x timesteps
                set_line.append((i, 1))
                self.disconnected_t[i] = 0

        redispatch = []
        curtail = []
        for gen in range(0, self._init_env.n_gen):

            if self.scaled:
                # if actions are scaled, convert the action to the original range
                # a = np.float32(
                #     (action[gen] + 1) / 2 * (self.gen_max[gen] - self.gen_min[gen])
                #     + self.gen_min[gen]
                # )
                a = np.interp(
                    action[gen], [-1, 1], [self.gen_min[gen], self.gen_max[gen]]
                )
            else:
                a = np.float32(action[gen])

            if self._init_env.gen_redispatchable[gen]:
                # if generator is redispatchable, add the action to the redispatch list
                redispatch.append((gen, a))
            else:
                # if generator is not redispatchable, add the action to the curtail list
                curtail.append(
                    (
                        gen,
                        np.float32(self.curtail_limit + (1 - self.curtail_limit) * a),
                    )
                )

        return self._init_env.action_space(
            {"set_line_status": set_line, "redispatch": redispatch, "curtail": curtail}
        )

    def update_curtail_limit(self, curtail_limit: float):
        """
        Update the curtailment limit during execution.

        :param curtail_limit: new curtailment limit
        """
        if not (0.0 <= curtail_limit <= 1.0):
            raise ValueError("Curtailment limit should be between 0 and 1")
        self.curtail_limit = curtail_limit

    def close(self):
        """
        Close the environment. For compatibility with sb3.
        """
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment
