import numpy as np
from matplotlib import pyplot
from env import StochasticMAB


# Use Bernoulli reward distribution, and Beta-Kernel for sampling
def thompson_sampling(total_time_slot, arm_num):
    bandit_model = StochasticMAB(n_arms=arm_num, random_type="Bernoulli")
    success = failures = [0 for i in range(arm_num)]
    alphas = betas = [1 for i in range(arm_num)]
    total_reward = [0]  # current total reward, recorded at each time slot

    for i in range(total_time_slot):
        # select arm
        thetas = [np.random.beta(success[j] + alphas[j], failures[j] + betas[j]) for j in range(arm_num)]
        arm = np.argmax(thetas)
        # roll
        this_reward = bandit_model.roll(arm)

        # update distribution
        if this_reward == 1:
            success[arm] += 1
        else:
            failures[arm] += 1
        total_reward.append(total_reward[-1] + this_reward)

    max_reward = bandit_model.max_expectation()
    regret = [i * max_reward - total_reward[i] for i in range(total_time_slot + 1)]
    return total_reward, regret


if __name__ == '__main__':
    sum_reward, cumulative_regret = thompson_sampling(total_time_slot=10000, arm_num=10)
    t = [100 * i for i in range(1, 100)]
    reward_t = [sum_reward[100 * i] for i in range(1, 100)]
    regret_t = [cumulative_regret[100 * i] for i in range(1, 100)]
    pyplot.plot(t, reward_t)
    pyplot.plot(t, regret_t)
    pyplot.show()
