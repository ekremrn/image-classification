import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


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


#
def plot_confusion_matrix(y_true, y_pred, classes, save_img):
    plt.figure(figsize = (15, 13), facecolor = 'silver', edgecolor = 'gray')

    cm = confusion_matrix(y_true, y_pred)

    if classes: cm = pd.DataFrame(cm, index = classes, columns = classes)
    
    sns.heatmap(cm, annot = True, fmt = '.3g')
    plt.xlabel("True")
    plt.ylabel("Pred")
    
    if save_img:
        plt.savefig(save_img)
    else:
        plt.show()


#
def plot_classification_report(y_true, y_pred, classes, save_img):
    plt.figure(figsize = (15, 15), facecolor = 'silver', edgecolor = 'gray')

    if classes:
        cr = classification_report(y_true, y_pred,
                                   target_names = classes,
                                   output_dict = True)
    else:
        cr = classification_report(y_true, y_pred,
                                   output_dict = True)
    
    sns.heatmap(pd.DataFrame(cr).iloc[:-1, :].T, annot=True)

    if save_img:
        plt.savefig(save_img)
    else:
        plt.show()
