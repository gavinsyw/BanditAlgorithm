import numpy as np


class DistributionError(Exception):
    def __init__(self, code=0x23550, message="Unsupported Distribution", args=("Unsupported Distribution",)):
        self.args = args
        self.code = code
        self.message = message


def generate_random_variable(ave, distribution):
    if distribution == "Gaussian":
        return np.random.normal(ave, 1, size=1)
    elif distribution == "Uniform":
        return np.random.uniform(0, ave * 2, size=1)
    elif distribution == "Bernoulli":
        rand = np.random.random_sample()
        if rand > ave:
            return 0
        else:
            return 1
    elif distribution == "Exponential":
        return np.random.exponential(1 / ave)


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
        return generate_random_variable(ave, self.distribution)

    def max_expectation(self):
        return max(self.ave)


class CombinatorialSemiMAB:
    def __init__(self, n_arms, max_arms, random_type="Gaussian"):
        possible_distribution = ["Gaussian", "Uniform", "Bernoulli", "Exponential"]
        self.n = n_arms
        self.max_arm = max_arms
        self.ave = [np.random.random_sample() for i in range(n_arms)]
        self.distribution = random_type
        if self.distribution not in possible_distribution:
            raise DistributionError

    def roll(self, arm_vec):
        arm_num = sum(arm_vec)
        if arm_num > self.max_arm:
            return 0
        return [generate_random_variable(self.ave[i], self.distribution)*arm_vec[i] for i in range(self.n)]

    def max_expectation(self):
        sorted_ave = np.sort(self.ave)
        return sum(sorted_ave[:-self.max_arm])
