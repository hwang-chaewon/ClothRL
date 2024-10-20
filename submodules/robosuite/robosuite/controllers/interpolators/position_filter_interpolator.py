from robosuite.controllers.interpolators.base_interpolator import Interpolator
import numpy as np
import robosuite.utils.transform_utils as T


class FilterInterpolator(Interpolator):
    """
    Simple class for implementing a linear interpolator.

    Abstracted to interpolate n-dimensions

    Args:
        ndim (int): Number of dimensions to interpolate

        controller_freq (float): Frequency (Hz) of the controller

        policy_freq (float): Frequency (Hz) of the policy model

        ramp_ratio (float): Percentage of interpolation timesteps across which we will interpolate to a goal position.

            :Note: Num total interpolation steps will be equal to np.floor(ramp_ratio * controller_freq / policy_freq)
                    i.e.: how many controller steps we get per action space update

        ori_interpolate (None or str): If set, assumes that we are interpolating angles (orientation)
            Specified string determines assumed type of input:

                `'euler'`: Euler orientation inputs
                `'quat'`: Quaternion inputs
    """

    def __init__(self,
                 ndim=6,
                 policy_freq=10,
                 controller_freq=100,
                 real_controller_freq=1000,
                 ramp_ratio=0.03,
                 ):
        # Number of dimensions to interpolate
        self.dim = ndim
        self.between_steps = int(real_controller_freq / controller_freq)
        self.ramp_ratio = ramp_ratio

    def set_goal(self, target):
        """
        Takes a requested (absolute) goal and updates internal parameters for next interpolation step

        Args:
            np.array: Requested goal (absolute value). Should be same dimension as self.dim
        """
        if target.shape[0] != self.dim:
            print("Requested goal: {}".format(goal))
            raise ValueError("FIlterInterpolator: Input size wrong for goal; got {}, needs to be {}!".format(
                goal.shape[0], self.dim))
        # Update start and goal
        self.start = self.target.copy()
        self.current = self.target.copy()
        self.target = np.array(target)

    def get_interpolated_goal(self):
        """
        Provides the next step in interpolation given the remaining steps.

        NOTE: If this interpolator is for orientation, it is assumed to be receiving either euler angles or quaternions

        Returns:
            np.array: Next position in the interpolated trajectory
        """
        for _ in range(self.between_steps):
            self.current = self.target*self.ramp_ratio + \
                (1-self.ramp_ratio)*self.current

        # Return the new interpolated step
        return self.current[:3], self.current[3:]

    def set_states(self, dim=None, ori=None, start=None):
        self.dim = dim if dim is not None else self.dim
        self.start = start
        self.current = start
        self.target = start
