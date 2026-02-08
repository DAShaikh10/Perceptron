"""
@Author: DAShaikh10
"""

import numpy as np


class HeavisideStepActivation:
    """
    Implements the Heaviside step function.

    `f(x) = 1 if x >= 0`
    `f(x) = 0 if x < 0`

    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/300px-Dirac_distribution_CDF.svg.png)
    """

    @staticmethod
    def activate(x: np.int8) -> np.uint8:
        """
        Applies the Heaviside step function.

        Args:
            x (np.int8): The input value (e.g., weighted sum).

        Returns:
            np.uint8: 0 or 1.
        """

        return np.uint8(x >= 0)
