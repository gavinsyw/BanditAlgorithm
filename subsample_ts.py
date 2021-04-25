import numpy as np
from matplotlib import pyplot
from env import StochasticMAB


# Use Bernoulli reward distribution, and Beta-Kernel for sampling
def subsample_ts(total_time_slot, arm_num, seed=10):
    distribution = "Gaussian"
    bandit_model = StochasticMAB(n_arms=arm_num, random_type=distribution)
    total_reward = [0]  # current total reward, recorded at each time slot
    ave_reward = list()     # average reward for each arm
    roll_time = list()      # roll time for each arm

    # subsample the arm set by the number of square root (total_time_slot)
    subsample_arm_num = int(np.sqrt(total_time_slot))
    subsample_arm_set = np.random.randint(0, arm_num, subsample_arm_num)
    sampled_values = [[] for i in range(subsample_arm_num)]

    for i in range(subsample_arm_num):
        current_arm = subsample_arm_set[i]
        this_reward = bandit_model.roll(current_arm)
        sampled_values[i].append(this_reward)
        ave_reward.append(this_reward)
        roll_time.append(1)
        total_reward.append(total_reward[-1]+this_reward)

    for i in range(total_time_slot-subsample_arm_num):
        # select arm
        thetas = [np.random.normal(np.average(sampled_values[j]), np.var(sampled_values[j])) for j in range(subsample_arm_num)]
        arm = np.argmax(thetas)
        # roll
        current_arm = subsample_arm_set[arm]
        this_reward = bandit_model.roll(current_arm)

        # update distribution
        sampled_values[arm].append(this_reward)
        total_reward.append(total_reward[-1] + this_reward)

    max_reward = bandit_model.max_expectation()
    regret = [i * max_reward - total_reward[i] for i in range(total_time_slot + 1)]
    return ave_reward, roll_time, total_reward, regret


if __name__ == '__main__':
    ave_reward, roll_time, sum_reward, cumulative_regret = subsample_ts(total_time_slot=500, arm_num=100)
    t = [i for i in range(1, 500)]
    reward_t = [sum_reward[i] for i in range(1, 500)]
    regret_t = [cumulative_regret[i] for i in range(1, 500)]
    # pyplot.plot(t, reward_t)
    pyplot.plot(t, regret_t)
    pyplot.show()
