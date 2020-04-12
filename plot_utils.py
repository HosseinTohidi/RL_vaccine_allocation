# from myPackage.read_file import read_file
from collections import defaultdict

import numpy as np
import pandas as pd

action_dict = defaultdict()
import matplotlib.pyplot as plt



def myPlot(states, time_list, groups_num, plot_shape=[]):
    s0, u0, e0, v0, i0, r0 = [], [], [], [], [], []
    for i in range(len(states)):
        state = np.array(states[i]).reshape(groups_num, 6)
        s0.append(state[:, 0].flatten())
        u0.append(state[:, 1].flatten())
        e0.append(state[:, 2].flatten())
        v0.append(state[:, 3].flatten())
        i0.append(state[:, 4].flatten())
        r0.append(state[:, 5].flatten())

    ds = pd.DataFrame(s0)
    de = pd.DataFrame(e0)
    di = pd.DataFrame(i0)
    dr = pd.DataFrame(r0)
    plt.close()
    if plot_shape != []:
        nrows, ncols = plot_shape[0], plot_shape[1]
    else:
        nrows, ncols = groups_num, 1
    fig, axes = plt.subplots(nrows, ncols)
    plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    for i in range(groups_num):
        dff = pd.DataFrame(data=[time_list, ds[i], de[i], di[i], dr[i]]).T
        dff.columns = ['time', 'S', 'E', 'I', 'R']
        if nrows == 1 or ncols == 1:
            dff.plot(x='time', ax=axes[i], legend=False)
            axes[i].set_title(f'age group{i + 1}')
            handles, labels = axes[groups_num - 1].get_legend_handles_labels()

        else:
            dff.plot(x='time', ax=axes[i // ncols][i % ncols], legend=False)
            axes[i // ncols][i % ncols].set_title(f'age group{i + 1}')
            handles, labels = axes[nrows - 1][ncols - 1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.25, 0.01, 0.5, .2), loc=3,
               ncol=4, mode="expand", borderaxespad=0.)
    # fig.delaxes(axes[1][2])
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=.05)
    plt.show()
