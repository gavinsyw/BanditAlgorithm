import numpy as np


class DistributionError(Exception):
    def __init__(self, code=0x23550, message="Unsupported Distribution", args=("Unsupported Distribution",)):
        self.args = args
        self.code = code
        self.message = message


class StochasticMAB:
    def __init__(self, n_arms, random_type="Gaussian"):
        possible_distribution = ["Gaussian", "Uniform", "Bernoulli", "Exponential"]
        self.n = n_arms
        self.ave = [np.random.random_sample() for i in range(n_arms)]
        self.distribution = random_type
        if self.distribution not in possible_distribution:
            raise DistributionError

    def roll(self, arm_num):
        ave = self.ave[arm_num]
        if self.distribution == "Gaussian":
            return np.random.normal(ave, 1, size=1)
        elif self.distribution == "Uniform":
            return np.random.uniform(0, ave*2, size=1)
        elif self.distribution == "Bernoulli":
            rand = np.random.random_sample()
            if rand > ave:
                return 0
            else:
                return 1
        elif self.distribution == "Exponential":
            return np.random.exponential(1/ave)

    def max_expectation(self):
        return max(self.ave)
