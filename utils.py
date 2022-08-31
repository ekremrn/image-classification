import numpy as np
import matplotlib.pyplot as plt


#
def graph_loss_acc(epoch, loss_l, acc_l, save_img = ""):

    plt.figure(figsize = [10, 12])

    plt.subplot(2, 1, 1)
    plt.plot(list(range(epoch)), loss_l)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Loss', fontsize = 12)

    plt.subplot(2, 1, 2)
    plt.plot(list(range(epoch)), acc_l)
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Acc %', fontsize = 12)

    if save_img:
        plt.savefig(save_img)
    else:
        plt.show()


#
def simple_graph(x, data, x_label = "", y_label = "", save_img = ""):

    plt.figure(figsize = [10, 7])
    plt.plot(list(range(x)), data)
    plt.xlabel(x_label, fontsize = 12)
    plt.ylabel(y_label, fontsize = 12)

    if save_img:
        plt.savefig(save_img)
    else:
        plt.show()
