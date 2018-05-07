import gym 
import HER.envs
import pandas as pd
import csv
import numpy as np

if __name__ == '__main__':
    env = gym.make("Reacher3d-v0")
    target_min = env.target_range_min
    target_max = env.target_range_max

    datapoints = int(1e4)

    out_csv_filename = "/tmp/suc/Reacher3d-v0_neg.csv"

    with open(out_csv_filename, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for _ in range(datapoints):
            ob = env.reset()

            in_range = True

            target_pos = env.np_random.uniform(-1, 1, size=3)
            while(not in_range):
                target_pos = env.np_random.uniform(-1, 1, size=3)

                if(np.all((target_pos - target_min)>0. and (target_pos - target_max)<0.)):
                    in_range = True

            
            writer.writerow(np.concatenate((ob[:-3], target_pos)))



