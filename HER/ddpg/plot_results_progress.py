import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
from ipdb import set_trace

if __name__ == "__main__":
	dirname = argv[1]

	print("checking the dir %s"%dirname)
	try:
		data = pd.read_csv(dirname + "progress.csv")
		eval_data = data["eval/return_history"]

		plt.plot(data["total/epochs"], data["eval/return_history"], label='eval')
		plt.plot(data["total/epochs"], data["rollout/return_history"], label='train')
		plt.legend()
		plt.xlabel('Epochs --->')
		plt.ylabel('Episode Reward ---->')
		plt.show()

		print("last reward in train:",data["rollout/return_history"][-1])

	except Exception as e:
		print(e)

