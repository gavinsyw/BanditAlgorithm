from subsample_ts import subsample_ts
from subsample_greedy import subsample_greedy
from subsample_ucb import subsample_ucb
from matplotlib import pyplot
import numpy as np

if __name__ == '__main__':
    iter_num = 40
    total_time_slot = 1000
    arm_num = 100
    reward_ucb = []
    reward_greedy = []
    reward_ts = []

    for iter_idx in range(iter_num):    
        x, y, greedy_reward, greedy_cumulative = subsample_greedy(total_time_slot=total_time_slot, arm_num=arm_num, seed=iter_idx)
        x, y, ucb_reward, ucb_cumulative = subsample_ucb(total_time_slot=total_time_slot, arm_num=arm_num, seed=iter_idx)
        x, y, ts_reward, ts_cumulative = subsample_ts(total_time_slot=total_time_slot, arm_num=arm_num, seed=iter_idx)
        reward_greedy.append(greedy_reward)
        reward_ucb.append(ucb_reward)
        reward_ts.append(ts_reward)
        # pyplot.plot(t, reward_t)

    t = [i for i in range(1, total_time_slot)]
    ave_reward_ucb = []
    ave_reward_greedy = []
    ave_reward_ts = []
    reward_greedy = np.array(reward_greedy)
    reward_ucb = np.array(reward_ucb)
    reward_ts = np.array(reward_ts)

    for i in range(1, total_time_slot):
        ave_reward_ucb.append(np.mean(reward_ucb[:,i]))
        ave_reward_greedy.append(np.mean(reward_greedy[:,i]))
        ave_reward_ts.append(np.mean(reward_ts[:,i]))

    pyplot.plot(t, ave_reward_greedy)
    pyplot.plot(t, ave_reward_ucb)
    pyplot.plot(t, ave_reward_ts)
    pyplot.legend(["greedy", "ucb", "ts"])
    pyplot.show()