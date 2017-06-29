import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_graph(G, all_edges, all_edges_with_weight, filename):
    pos = nx.spring_layout(G, k=0.8, iterations=20)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, linewidths=0)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, width=0.2)
    edge_labels = {}
    for index, edge in enumerate(all_edges_with_weight):
        edge_labels[(edge[0], edge[1])] = all_edges_with_weight[index][2]["weight"]
    nx.draw_networkx_edge_labels(G, pos, font_size=4, edge_labels=edge_labels)

    # labels
    nx.draw_networkx_labels(G, pos, font_size=6, font_family='sans-serif')

    plt.axis('off')

    print("saving to %s" % (filename))
    plt.savefig(filename, dpi=800)  # save as png
    # plt.show()  # display
    plt.clf()

def plot_pie_chart(labels, sizes, filename, explode=None):
    plt.clf()

    plt.rcParams['font.size'] = 6.0

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, shadow=False, startangle=90) #, autopct='%1.1f%%')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # texts[0].set_fontsize(1)

    print("saving to %s" % (filename))
    plt.savefig(filename, dpi=800)  # save as png
    plt.clf()


def autolabel(ax, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


def plot_bar_chart(title, labels, labels_data, labels_description, y_label, filename, width=0.35, labels_std=None):

    plt.clf()

    N = len(labels)
    men_means = tuple(labels_data)
    if not labels_std:
        labels_std = [0] * len(labels)
    men_std = tuple(labels_std)

    # Get current size
    fig_size = plt.rcParams["figure.figsize"]

    # Prints: [8.0, 6.0]
    #print("Current size: %s" % (fig_size))

    # Set figure width to 12 and height to 9
    fig_size[0] = 4 * len(labels)
    fig_size[1] = 2 * len(labels)
    plt.rcParams["figure.figsize"] = fig_size



    ind = np.arange(N)  # the x locations for the groups

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)
    ax.tick_params(labelsize=15, direction="in")
    # women_means = (25, 32, 34, 20, 25)
    # women_std = (3, 5, 2, 3, 3)
    # rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

    # add some text for labels, title and axes ticks
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tuple([x[::-1] for x in labels]))

    ax.legend((rects1[0],), (labels_description,))

    autolabel(ax, rects1)
    # autolabel(rects2)

    # plt.show()

    print("saving to %s" % (filename))
    plt.savefig(filename, dpi=800)  # save as png
    plt.clf()