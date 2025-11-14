class HeavisideStepActivation:
    """
    Implements the Heaviside step function.

    `f(x) = 1 if x >= 0, 0 if x < 0`

    ![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Dirac_distribution_CDF.svg/300px-Dirac_distribution_CDF.svg.png)
    """

    @staticmethod
    def activate(x: float) -> float:
        """
        Applies the Heaviside step function.

        Args:
            x (float): The input value (e.g., weighted sum).

        Returns:
            int: 0 or 1.
        """

        return float(x >= 0.0)
