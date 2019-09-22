import numpy as np
from matplotlib import pyplot
from env import CombinatorialSemiMAB
import math


# Supported random type: Gaussian, Uniform, Bernoulli, Exponential
def combinatorial_UCB(total_time_slot, arm_num, max_arm):
    bandit_model = CombinatorialSemiMAB(n_arms=arm_num, max_arms=max_arm, random_type="Gaussian")
    ave_reward = list()  # average reward for each arm
    roll_time = list()  # roll time for each arm
    total_reward = [0]  # current total reward, recorded at each time slot

    # roll each arm once first
    for i in range(arm_num):
        this_reward = bandit_model.roll([0]*i+[1]+[0]*(arm_num-i-1))
        ave_reward.append(sum(this_reward))
        roll_time.append(1)
        total_reward.append(total_reward[-1] + sum(this_reward))

    for i in range(total_time_slot - arm_num):
        reward_hat = [ave_reward[j]+math.sqrt(2*math.log(1+i)/roll_time[j]) for j in range(arm_num)]
        arms = oracle(reward_hat, max_arm, arm_num)

        this_reward = bandit_model.roll(arms)
        ave_reward = [(ave_reward[arm] * roll_time[arm] + this_reward[arm]) / (roll_time[arm] + arms[arm]) for arm in range(arm_num)]
        roll_time = [roll_time[arm] + arms[arm] for arm in range(arm_num)]
        total_reward.append(total_reward[-1] + sum(this_reward))

    max_reward = bandit_model.max_expectation()
    regret = [i*max_reward-total_reward[i] for i in range(total_time_slot+1)]
    return ave_reward, roll_time, total_reward, regret


def oracle(reward_hat, max_arm, arm_num):
    arms = [0 for i in range(arm_num)]
    sorted_reward = np.sort(reward_hat)
    for r in sorted_reward[-max_arm:]:
        if r in reward_hat:
            arms[reward_hat.index(r)] = 1
    return arms


if __name__ == '__main__':
    a_reward, r_time, sum_reward, cumulative_regret = combinatorial_UCB(total_time_slot=10000, arm_num=10, max_arm=3)
    t = [100*i for i in range(1, 100)]
    reward_t = [sum_reward[100*i] for i in range(1, 100)]
    regret_t = [cumulative_regret[100*i] for i in range(1, 100)]
    pyplot.plot(t, reward_t)
    pyplot.plot(t, regret_t)
    pyplot.show()
