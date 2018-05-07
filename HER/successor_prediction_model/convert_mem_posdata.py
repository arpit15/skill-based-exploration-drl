import numpy as np
import argparse

def convert(log_dir, in_shape):
	in_csv_filename = osp.join(log_dir, "%s.csv"%env_id)
	out_csv_filename = osp.join(log_dir, "%s_pos.csv"%env_id)

	# search if the file already exists
    if not osp.exists(csv_filename):
        print("File is not present!")
        return False

    base_dataset = np.loadtxt(in_csv_filename, delimiter= ',')
    state = base_dataset[:, :in_shape]

    labels = np.ones((state.shape[0],1))
    pos_dataset = np.concatenate((state, labels))

    numpy.savetxt(out_csv_filename, pos_dataset, delimiter=",")
    print("File %s written"%out_csv_filename)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--log-dir', type=str, default='/tmp/her')
    parser.add_argument('--in-shape', type=int, default=2000)
    
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    run(**args)
