from cProfile import label
import pickle
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    epochs = [_ for _ in range(1, 101)]
    directory_str = './accuracy_results'
    directory = os.fsencode(directory_str)

    mode_to_acc = {}
    indexed_mode = ['vanilla', 'vanilla_topk', 'gauss_sgd', 'gauss_sgd_topk',  'gauss_sgd_topk_accum', ]

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        with open (directory_str + '/' + filename, 'rb') as fp:
            mode_to_acc[filename] = pickle.load(fp)
            plt.plot(epochs, mode_to_acc[filename], label=filename)
    plt.legend(loc="lower center")
    plt.savefig('./plots/accuracy_curves_new.png')
