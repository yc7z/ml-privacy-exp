from cProfile import label
import pickle
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    epochs = [_ for _ in range(1, 61)]
    directory_str = './accumulate_momentum_v1_acc'
    directory = os.fsencode(directory_str)

    mode_to_acc = {}

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        with open (directory_str + '/' + filename, 'rb') as fp:
            mode_to_acc[filename] = pickle.load(fp)
            acc_lst = []
            for t in mode_to_acc[filename]:
                acc_lst.append(t.item())
            plt.plot(epochs, acc_lst, label=filename)
    plt.legend(loc="lower center", bbox_to_anchor=(0.8, 0.25))
    plt.savefig('./plots_new/accumulate_momentum_acc_v1(2).png')
