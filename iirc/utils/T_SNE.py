import numpy as np
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import torch

import pdb

def draw_tsne(X, labels, tag, show_number=50, output_dir=None, title=None):
  #  X, labels = shuffle(X, labels)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, labels, tag, show_number=show_number, output_dir=output_dir, title=title)


def get_adaption_X(X_vector):
    row_vector = X_vector[:, 0]
    col_vector = X_vector[:, 1]
    number_num = len(row_vector)
    q1 = int(1 * (number_num + 1) / 4)
    q3 = int(3 * (number_num + 1) / 4)
    new_vector = np.sort(row_vector)
    try:
        num_q1 = new_vector[q1]
    except:
        pdb.set_trace()
    num_q3 = new_vector[q3]
    bottom_row = num_q1 - 1.5 * (num_q3 - num_q1)
    top_row = num_q3 + 1.5 * (num_q3 - num_q1)

    new_vector = np.sort(col_vector)
    num_q1 = new_vector[q1]
    num_q3 = new_vector[q3]
    bottom_col = num_q1 - 1.5 * (num_q3 - num_q1)
    top_col = num_q3 + 1.5 * (num_q3 - num_q1)

    new_X = np.array([0, 0])
    for i in range(number_num):
        if X_vector[i, 0] < top_row and X_vector[i, 0] > bottom_row and X_vector[i, 1] < top_col and X_vector[i, 1] > bottom_col:
            new_X = np.vstack((new_X, X_vector[i, :]))
    return new_X


def plot_embedding(X, labels, tag, show_number=50, output_dir=None, title=None):
    # tag表示的是 labels里面的标签标号对应的标签意义
    #  x_min, x_max = np.min(X, 0), np.max(X, 0)
    #  X = (X - x_min) / (x_max - x_min)
    X = np.array(X)
    labels = np.array(labels)
    plt.figure()
    color_set = ['r', 'c', 'plum', 'g', 'b', 'y', 'm', 'gray', 'salmon', 'sienna', 'indigo', 'k', 'brown', 'olive', 'deepskyblue',  'sienna', 'coral',  'orchid'] * 8

  #  pdb.set_trace()
    c_max = np.max(labels) + 1
    color_index = 0
    for i in range(c_max):
        temp_x = np.where(labels == i)[0]
        if len(temp_x) == 0:
            continue
        if len(temp_x) > show_number:
            random_index = torch.randperm(len(temp_x))
            temp_x = temp_x[random_index[:show_number]]
        new_X = get_adaption_X(X[temp_x, :])
        plt.scatter(new_X[:, 0], new_X[:, 1], c=color_set[color_index], marker='o')
        color_index = color_index + 1
         #   print(f'{c_max}_{i}:{color_set[i]}')
    plt.legend(tag, loc=(1.1, 0.5))
    plt.title(title)
    plt.show()
   # pdb.set_trace()
    if (output_dir is not None):
        plt.savefig(f'{output_dir}/{title}.png', bbox_inches='tight', dpi=500)
    plt.pause(1)
    plt.close()