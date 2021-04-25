import numpy as np
from matplotlib import pyplot
from env import StochasticMAB
import random


# Supported random type: Gaussian, Uniform, Bernoulli, Exponential
def subsample_greedy(total_time_slot, arm_num, seed=10):
    bandit_model = StochasticMAB(n_arms=arm_num, random_type="Gaussian")
    ave_reward = list()     # average reward for each arm
    roll_time = list()      # roll time for each arm
    total_reward = [0]   # current total reward, recorded at each time slot

    # subsample the arm set by the number of square root (total_time_slot)
    subsample_arm_num = int(np.sqrt(total_time_slot))
    subsample_arm_set = np.random.randint(0, arm_num, subsample_arm_num)

    # roll each arm once first
    for i in range(subsample_arm_num):
        current_arm = subsample_arm_set[i]
        this_reward = bandit_model.roll(current_arm)
        ave_reward.append(this_reward)
        roll_time.append(1)
        total_reward.append(total_reward[-1]+this_reward)

    for i in range(total_time_slot - subsample_arm_num):
        arm = np.argmax(ave_reward)
        current_arm = subsample_arm_set[arm]

        this_reward = bandit_model.roll(current_arm)
        ave_reward[arm] = (ave_reward[arm] * roll_time[arm] + this_reward) / (roll_time[arm] + 1)
        roll_time[arm] += 1
        total_reward.append(total_reward[-1] + this_reward)

    max_reward = bandit_model.max_expectation()
    regret = [i*max_reward-total_reward[i] for i in range(total_time_slot+1)]
    return ave_reward, roll_time, total_reward, regret


if __name__ == '__main__':
    a_reward, r_time, sum_reward, cumulative_regret = subsample_greedy(total_time_slot=500, arm_num=100)
    t = [i for i in range(1, 500)]
    reward_t = [sum_reward[i] for i in range(1, 500)]
    regret_t = [cumulative_regret[i] for i in range(1, 500)]
    pyplot.plot(t, reward_t)
    pyplot.plot(t, regret_t)
    pyplot.show()
