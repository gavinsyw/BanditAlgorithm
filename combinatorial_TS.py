import numpy as np
from matplotlib import pyplot
from env import CombinatorialSemiMAB
import math


# Supported random type: Gaussian, Uniform, Bernoulli, Exponential
def combinatorial_TS(total_time_slot, arm_num, max_arm):
    bandit_model = CombinatorialSemiMAB(n_arms=arm_num, max_arms=max_arm, random_type="Bernoulli")
    success = failures = [0 for i in range(arm_num)]
    alphas = betas = [1 for i in range(arm_num)]
    total_reward = [0]  # current total reward, recorded at each time slot

    for i in range(total_time_slot):
        # Update thetas
        thetas = [np.random.beta(success[j] + alphas[j], failures[j]+betas[j]) for j in range(arm_num)]
        arms = oracle(thetas, max_arm, arm_num)

        this_reward = bandit_model.roll(arms)
        for n in range(arm_num):
            if this_reward[n] == 1 and arms[n] == 1:
                success[n] += 1
            elif this_reward[n] == 0  and arms[n] == 1:
                failures[n] += 1
        total_reward.append(total_reward[-1] + sum(this_reward))

    max_reward = bandit_model.max_expectation()
    regret = [i*max_reward-total_reward[i] for i in range(total_time_slot+1)]
    return total_reward, regret


def oracle(reward_hat, max_arm, arm_num):
    arms = [0 for i in range(arm_num)]
    sorted_reward = np.sort(reward_hat)
    for r in sorted_reward[-max_arm:]:
        if r in reward_hat:
            arms[reward_hat.index(r)] = 1
    return arms


if __name__ == '__main__':
    sum_reward, cumulative_regret = combinatorial_TS(total_time_slot=10000, arm_num=10, max_arm=3)
    t = [100*i for i in range(1, 100)]
    reward_t = [sum_reward[100*i] for i in range(1, 100)]
    regret_t = [cumulative_regret[100*i] for i in range(1, 100)]
    pyplot.plot(t, reward_t)
    pyplot.plot(t, regret_t)
    pyplot.show()
