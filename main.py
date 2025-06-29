from lib.utils import preturb
import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_datasets():
    with open('dataset/TRAIN.pkl', 'rb') as f:
        data = pickle.load(f)

    return data

def get_sample(data, idx):
    svara, form, prec, curr, succ = data[idx]

    return prec, curr, succ

def plot_ts(prec, curr, succ, aug=False):
    color = 'r' if aug else 'g'
    label = 'Augmented' if aug else 'Original'

    seq = np.concatenate([prec, curr, succ])
    times = np.arange(len(seq)) * 0.0044

    plt.plot(times, seq, c=color, label=label)

def main():
    data = load_datasets()

    idx = np.random.randint(len(data))

    prec, curr, succ = get_sample(data, idx)

    prec_new = preturb(prec)
    curr_new = preturb(curr)
    succ_new = preturb(succ)

    plot_ts(prec, curr, succ)
    plot_ts(prec_new, curr_new, succ_new, aug=True)

    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (cents)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
