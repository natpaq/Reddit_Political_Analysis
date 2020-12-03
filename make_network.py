import networkx as nx
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter as ctr
from networkx.algorithms import community as com
import networkx.algorithms.community as nx_comm
import matplotlib as mpl
import matplotlib.cm as cm
import itertools
import sys
from community import community_louvain
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score


def get_common_subreddits(data, start, end):
    common = 0
    for subreddit in data[start]:
        if subreddit in data[end]:
            common += 1
    return common


def user_network(biden_data, trump_data, title):
    G = nx.Graph()
    # chop down the graph for it to make more sense
    # data = {k: data[k] for k in list(data)[:10]}
    # First add the nodes. Each node is the individual. also add their ground truth
    for user in biden_data.keys():
        G.add_node(user, value="Biden")
    for user in trump_data.keys():
        G.add_node(user, value="Trump")
    # merge the dictionaries
    data = {**biden_data, **trump_data}
    # Next add edges. Each edge between individuals indicates common subreddits.
    for start in G.nodes():
        for end in G.nodes():
            if start != end and not (G.has_edge(start, end) or G.has_edge(end, start)):
                # make edge the common users between these nodes
                common = get_common_subreddits(data, start, end)
                if common > 0:
                    G.add_edge(start, end, weight=common)
    # save to view in gephi
    nx.write_gexf(G, "./user_data/graphs/" + title +"_graph.gexf")
    return G


def supporter_user_network(data, label, title):
    G = nx.Graph()
    # chop down the graph for it to make more sense
    # data = {k: data[k] for k in list(data)[:10]}
    # First add the nodes. Each node is the individual. also add their ground truth
    for user in data.keys():
        G.add_node(user, value="Biden")
    # Next add edges. Each edge between individuals indicates common subreddits.
    for start in G.nodes():
        for end in G.nodes():
            if start != end and not (G.has_edge(start, end) or G.has_edge(end, start)):
                # make edge the common users between these nodes
                common = get_common_subreddits(data, start, end)
                if common > 0:
                    G.add_edge(start, end, weight=common)
    # save to view in gephi
    nx.write_gexf(G, "./user_data/graphs/" + title +"_graph.gexf")
    return G


def get_common_users(data, start, end):
    total_users = 0
    for user, subreddits in data.items():
        if start in subreddits and end in subreddits:
            total_users += 1
    return total_users


def subreddit_network(data, title):
    G = nx.Graph()
    # chop down the graph for it to make more sense
    # data = {k: data[k] for k in list(data)[:3]}
    # First add the nodes. Each node is a subreddit
    for user, subreddits in data.items():
        for subreddit in subreddits:
            G.add_node(subreddit)
    # now add edges, edge is a user active in both
    for start in G.nodes():
        for end in G.nodes():
            if start != end and not (G.has_edge(start, end) or G.has_edge(end, start)):
                # make edge the common users between these nodes
                common = get_common_users(data, start, end)
                if common > 0:
                    G.add_edge(start, end, weight=common)
    # save to view in gephi
    nx.write_gexf(G, "./user_data/graphs/" + title + "_subreddit_graph.gexf")
    return G


def print_girvan_newman_graph(G):
    k = 2
    count = 0
    inc = 0
    node_communities = dict()
    communities_generator = com.girvan_newman(G)

    for communities in itertools.islice(communities_generator, k):
        inc += 1
        if inc == k:
            for community in communities:
                count += 1
                for node in community:
                    node_communities[node] = count

    ## Calculate modularity
    mod = nx_comm.modularity(G, next(communities_generator))
    print("Modularity:", mod)

    nx.draw(G, node_color=list(node_communities.values()))  # ,with_labels=True)
    plt.show()

    values = dict()
    i = 0
    for node, data in G.nodes(data=True):
        i += 1
        values[i] = data['value']

    labels_true = list(values.values())
    labels_pred = list(node_communities.values())  # i hope this is it

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    print('NMI: {:.4f}'.format(nmi))

    ari = adjusted_rand_score(labels_true, labels_pred)
    print('ARI: {:.4f}'.format(ari))


def print_louvain_algo(g):
    # First partition the graph according to the Louvain Algorithm
    partition = community_louvain.best_partition(g)

    # start making a new graph that we will plot with the partition information gathered from community_louvain
    # communities = set(partition.values())

    # print modularity
    mod = community_louvain.modularity(partition, g)
    print("Modularity:", mod)

    # Draw up the graph
    pos = nx.spring_layout(g)
    nx.draw(g, node_color=list(partition.values()))
    plt.show()

    # Extract ground truth labels
    values = dict()
    i = 0
    for node, data in g.nodes(data=True):
        i += 1
        values[i] = data['value']

    labels_true = list(values.values())
    # Labels to evaluate against
    labels_pred = list(partition.values())

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    print('NMI: {:.4f}'.format(nmi))

    ari = adjusted_rand_score(labels_true, labels_pred)
    print('ARI: {:.4f}'.format(ari))


def make_and_analyze_user_network(biden_data, trump_data):
    g = user_network(biden_data, trump_data, "All_Users")
    print_louvain_algo(g)
    g = supporter_user_network(biden_data, "Biden", "Biden_User")
    print_louvain_algo(g)
    g = supporter_user_network(trump_data, "Trump", "Trump_User")
    print_louvain_algo(g)


def make_and_analyze_subreddit_networks(biden_data, trump_data):
    G_biden = subreddit_network(biden_data, "Biden")
    G_trump = subreddit_network(biden_data, "Trump")


if __name__ == '__main__':
    with open('./user_data/biden_user_subred_freq.json') as json_file:
        biden_data = json.load(json_file)
    with open('./user_data/trump_user_subred_freq.json') as json_file:
        trump_data = json.load(json_file)
    make_and_analyze_user_network(biden_data, trump_data)



