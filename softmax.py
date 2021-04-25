import numpy as np
from matplotlib import pyplot
from env import StochasticMAB
import math


# Supported random type: Gaussian, Uniform, Bernoulli, Exponential
def softmax(total_time_slot, arm_num, tau):
    bandit_model = StochasticMAB(n_arms=arm_num, random_type="Gaussian")
    ave_reward = list()  # average reward for each arm
    roll_time = list()  # roll time for each arm
    total_reward = [0]  # current total reward, recorded at each time slot

    # roll each arm once first
    for i in range(arm_num):
        this_reward = bandit_model.roll(i)
        ave_reward.append(this_reward)
        roll_time.append(1)
        total_reward.append(total_reward[-1] + this_reward)

    for i in range(total_time_slot - arm_num):
        p_arm_list = [math.exp(ave_reward[0])]
        for j in range(1, arm_num):
            p_arm = math.exp(ave_reward[j]/tau)+p_arm_list[-1]
            p_arm_list.append(p_arm)
        p_arm_list = [k / p_arm_list[-1] for k in range(arm_num)]

        arm = -1
        r = np.random.random_sample()
        for j in range(arm_num-1):
            if r >= p_arm_list[j] and r < p_arm_list[j+1]:
                arm = j
                break
        if arm == -1:
            arm = arm_num - 1

        this_reward = bandit_model.roll(arm)
        ave_reward[arm] = (ave_reward[arm] * roll_time[arm] + this_reward) / (roll_time[arm] + 1)
        roll_time[arm] += 1
        total_reward.append(total_reward[-1] + this_reward)

    max_reward = bandit_model.max_expectation()
    regret = [i*max_reward-total_reward[i] for i in range(total_time_slot+1)]
    return ave_reward, roll_time, total_reward, regret


if __name__ == '__main__':
    a_reward, r_time, sum_reward, cumulative_regret = softmax(total_time_slot=10000, arm_num=10, tau=0.7)
    t = [100*i for i in range(1, 100)]
    reward_t = [sum_reward[100*i] for i in range(1, 100)]
    regret_t = [cumulative_regret[100*i] for i in range(1, 100)]
    pyplot.plot(t, reward_t)
    pyplot.plot(t, regret_t)
    pyplot.show()
