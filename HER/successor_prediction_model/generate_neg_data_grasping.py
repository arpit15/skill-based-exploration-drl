import gym 
import HER.envs
import pandas as pd
import csv

if __name__ == '__main__':
	env = gym.make("picknmovet-v2")
	
	datapoints = int(1e4)

	out_csv_filename = "/tmp/suc/grasping-v2_neg.csv"

	with open(out_csv_filename, 'w',newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            for _ in range(datapoints):
                ob = env.reset()
                writer.writerow(ob)



