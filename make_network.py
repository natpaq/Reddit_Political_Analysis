import networkx as nx
import json
import numpy as np
from random import sample
from scipy import stats
from scipy.sparse.csgraph import connected_components
import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from collections import Counter as ctr
from networkx.algorithms import community as com
import networkx.algorithms.community as nx_comm
from scipy.sparse import csr_matrix
import matplotlib as mpl
import matplotlib.cm as cm
import itertools
import sys
from community import community_louvain
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
#Complicated ML part for real-node-labels
import networkx as nx
import itertools
import json
import os
from sklearn.metrics.cluster import adjusted_mutual_info_score
import numpy as np
from networkx.algorithms import node_classification
from networkx.readwrite import json_graph
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import cdlib
from cdlib import algorithms
import stellargraph as sg
from stellargraph.data import BiasedRandomWalk
from stellargraph.mapper import ClusterNodeGenerator, StellarGraph
from stellargraph.layer import GCN, GAT
from stellargraph import globalvar
from tensorflow.keras import backend as K

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
from IPython.display import display, HTML


from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from stellargraph.layer import GCN

def get_nmi(labels_true, labels_pred):
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return nmi


def get_ari(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari


def do_Leiden(g):
    values0 = dict()
    values1 = dict()
    i = 0
    j = 0
    for node, data in g.nodes(data=True):
        if data['value'] == 0:
            values0[i] = data['value']
            i += 1
        elif data['value'] == 1:
            values1[i] = data['value']
            j += 1

    truth = cdlib.NodeClustering(communities=[list(values0), list(values1)], graph=g, method_name="louvain")
    lp_coms = algorithms.label_propagation(g)
    leiden_coms = algorithms.leiden(g)
    print("AMI")
    print(leiden_coms.adjusted_mutual_information(lp_coms))
    print("NMI")
    print(leiden_coms.normalized_mutual_information(lp_coms))


def do_node_perception(g):
    lp_coms = algorithms.label_propagation(g)
    node_coms = algorithms.node_perception(g, threshold=0.25, overlap_threshold=0.25)
    print(node_coms.normalized_mutual_information(lp_coms))
    print(node_coms.modularity_density())


def jaccard_weights(graph, _subjects, edges):
    sources = graph.node_features(edges.source)
    targets = graph.node_features(edges.target)

    intersection = np.logical_and(sources, targets)
    union = np.logical_or(sources, targets)

    return intersection.sum(axis=1) / union.sum(axis=1)


def do_node_2_vec(g):
    # works with weighted edges
    walk_length = 100  # maximum length of a random walk to use throughout this notebook
    subjects = pd.DataFrame(list(g.nodes()), nx.get_node_attributes(g, 'value').values())
    G = StellarGraph.from_networkx(g)

    rw = BiasedRandomWalk(G)
    weighted_walks = rw.run(
        nodes=G.nodes(),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    print("Number of random walks: {}".format(len(weighted_walks)))
    weighted_model = Word2Vec(
        weighted_walks, size=128, window=5, min_count=0, sg=1, workers=1, iter=1
    )
    # Retrieve node embeddings and corresponding subjects
    node_ids = weighted_model.wv.index2word  # list of node IDs
    weighted_node_embeddings = (
        weighted_model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    # the gensim ordering may not match the StellarGraph one, so rearrange
    node_targets = pd.DataFrame(nx.get_node_attributes(g, 'value').values()).astype("category")
    # Apply t-SNE transformation on node embeddings
    tsne = TSNE(n_components=2, random_state=42)
    weighted_node_embeddings_2d = tsne.fit_transform(weighted_node_embeddings)
    # draw the points
    alpha = 0.7

    plt.figure(figsize=(10, 8))
    plt.scatter(
        weighted_node_embeddings_2d[:, 0],
        weighted_node_embeddings_2d[:, 1],
        c=node_targets[0],
        cmap="jet",
        alpha=0.7,
    )
    plt.show()
    # X will hold the 128-dimensional input features
    X = weighted_node_embeddings
    # y holds the corresponding target values
    y = np.array(node_targets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, test_size=None, random_state=42
    )
    print(
        "Array shapes:\n X_train = {}\n y_train = {}\n X_test = {}\n y_test = {}".format(
            X_train.shape, y_train.shape, X_test.shape, y_test.shape
        )
    )
    clf = LogisticRegressionCV(
        Cs=10,
        cv=10,
        tol=0.001,
        max_iter=1000,
        scoring="accuracy",
        verbose=False,
        multi_class="ovr",
        random_state=5434,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))


def do_gat(g):
    features = []
    le = preprocessing.LabelEncoder()
    le.fit(g.nodes())
    lads = dict(zip(list(g.nodes()), le.transform(list(g.nodes()))))
    for node, data in g.nodes(data=True):
        features.append([data['value'], node])
    g = nx.relabel_nodes(g, lads)
    node_data = pd.DataFrame(nx.get_node_attributes(g, 'value').values(), index=list(lads.values()))
    # cannot do gat cause requires node features
    labels = pd.DataFrame(list(lads.values()), nx.get_node_attributes(g, 'value').values())
    G = StellarGraph.from_networkx(g, node_features=node_data)
    train_size = 140

    train_labels, test_labels = model_selection.train_test_split(
        labels, train_size=train_size, test_size=None,
    )
    val_subjects, test_subjects = model_selection.train_test_split(
        test_labels, train_size=430, test_size=2,
    )

    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_labels)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)

    generator = FullBatchNodeGenerator(G, method="gat")
    train_gen = generator.flow(train_labels.index, train_targets)
    gat = GAT(
        layer_sizes=[8, train_targets.shape[1]],
        activations=["elu", "softmax"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
    )
    x_inp, predictions = gat.in_out_tensors()
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    val_gen = generator.flow(val_subjects.index, val_targets)

    os.makedirs("log")
    es_callback = EarlyStopping(monitor="val_acc", patience=20)
    mc_callback = ModelCheckpoint(
        "log/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
    )

    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback, mc_callback],
    )
    sg.utils.plot_history(history)
    plt.show()


def merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination


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
        G.add_node(user, value=0)
    for user in trump_data.keys():
        G.add_node(user, value=1)
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


def global_subreddit_network(biden_data, trump_data, title):
    G = nx.Graph()

    for user, subreddits in trump_data.items():
        for subreddit in subreddits:
            G.add_node(subreddit, value=0)

    for user, subreddits in biden_data.items():
        for subreddit in subreddits:
            if G.has_node(subreddit):
                G.nodes[subreddit]["value"] = 3
            else:
                G.add_node(subreddit, value=1)

    # merge the dictionaries
    data = merge(biden_data, trump_data)

    # now add edges, edge is a user active in both
    for start in G.nodes():
        for end in G.nodes():
            if start != end and not (G.has_edge(start, end) or G.has_edge(end, start)):
                # make edge the common users between these nodes
                common = get_common_users(data, start, end)
                if common > 0:
                    G.add_edge(start, end, weight=common)
    # save to view in gephi
    nx.write_gexf(G, "./user_data/graphs/global_subreddit_graph.gexf")
    return G



def supporter_user_network(data, title):
    G = nx.Graph()
    # chop down the graph for it to make more sense
    # data = {k: data[k] for k in list(data)[:10]}
    # First add the nodes. Each node is the individual. also add their ground truth
    for user in data.keys():
        G.add_node(user, value=0)
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


def do_girvan_newman_graph(G):
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
        values[i] = data['value']
        i += 1


    labels_true = list(values.values())
    labels_pred = list(node_communities.values())  # i hope this is it

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    print('NMI: {:.4f}'.format(nmi))

    ari = adjusted_rand_score(labels_true, labels_pred)
    print('ARI: {:.4f}'.format(ari))

    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    print('AMI: {:.4f}'.format(ami))


def do_louvain_algo(g):
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
        values[node] = data['value']
        i += 1

    labels_true = list(values.values())
    # Labels to evaluate against
    labels_pred = list(partition.values())

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    print('NMI: {:.4f}'.format(nmi))

    ari = adjusted_rand_score(labels_true, labels_pred)
    print('ARI: {:.4f}'.format(ari))

    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    print('AMI: {:.4f}'.format(ami))


def run_classification(G, node_classification_function, graph_name):
    runs = 10

    # first get and set the node labels:
    for node, data in G.nodes(data=True):
        G.nodes[node]['label'] = data['value']

    num_of_nodes = len(G)

    labels_true = list(nx.get_node_attributes(G, 'label').values())

    # structures to hold the ari and nmi measures
    nmi_measures = []
    ari_measures = []
    avg_run_nmi = []
    avg_run_ari = []

    # run 10 times
    for i in reversed(range(5, 100, 10)):
        # randomly drop labels of nodes from 95% to 5%
        nmi = []
        ari = []
        for j in range(runs):
            # I do not want to destroy the original graph so we make a temporary graph
            temp = G.copy()

            # randomly select a sample of nodes
            random_nodes = sample(list(temp.nodes()), round((i / 100) * num_of_nodes))

            # and now drop these nodes labels!
            for node in random_nodes:
                temp.nodes[node]['label'] = 0

            # *predict* the labels
            labels_pred = node_classification_function(temp)

            # now measure the algo's accuracy to predict labels for overall average
            nmi_measures.append(get_nmi(labels_true, labels_pred))
            ari_measures.append(get_ari(labels_true, labels_pred))

            # these are for the plot
            nmi.append(get_nmi(labels_true, labels_pred))
            ari.append(get_ari(labels_true, labels_pred))
        # these are for the plot
        avg_run_nmi.append(sum(nmi) / runs)
        avg_run_ari.append(sum(ari) / runs)

    x_axis = [i for i in reversed(range(5, 100, 10))]

    # now average this over the total number of runs to get the accuracy of the function!
    # NMI
    print(node_classification_function.__name__ + " algorithm:")
    plt.plot(x_axis, avg_run_nmi, 'ro')
    plt.xlabel('Percentage of Labels Dropped (%)')
    plt.ylabel('Average NMI')
    plt.title('Node classification NMI accuracy of the ' + graph_name + " dataset")
    plt.show()
    average_nmi = sum(nmi_measures) / (10 * runs)
    print('Overall average NMI: {:.4f}'.format(average_nmi))
    # ARI
    plt.plot(x_axis, avg_run_ari, 'ro')
    plt.xlabel('Percentage of Labels Dropped (%)')
    plt.ylabel('Average ARI')
    plt.title('Node classification ARI accuracy of the ' + graph_name + " dataset")
    plt.show()
    average_ari = sum(ari_measures) / (10 * runs)
    print('Overall average ARI: {:.4f}'.format(average_ari))
    return average_nmi, average_ari


def print_kernighan_aglo(g):
    # First partition the graph according to the Louvain Algorithm
    partition = com.kernighan_lin.kernighan_lin_bisection(g)

    # Extract ground truth labels
    values = dict()
    i = 0
    for node, data in g.nodes(data=True):
        values[i] = data['value']
        i += 1

    pred = dict()
    i = 0
    for man in partition[0]:
        pred[i] = "Trump"
        i += 1
    for man in partition[1]:
        pred[i] = "Biden"
        i += 1

    pred = dict()
    i = 0
    for man in partition[0]:
        pred[i] = "Biden"
        i += 1
    for man in partition[1]:
        pred[i] = "Trump"
        i += 1

    labels_true = list(values.values())
    # Labels to evaluate against
    labels_pred = list(pred)

    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    print('NMI: {:.4f}'.format(nmi))

    ari = adjusted_rand_score(labels_true, labels_pred)
    print('ARI: {:.4f}'.format(ari))

    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    print('AMI: {:.4f}'.format(ami))


def get_top_five(node_dict):
    counter = ctr(node_dict)
    top = counter.most_common(5)
    return top


def top_5(g):
    g = g.to_directed()
    top_5_indeg = get_top_five(nx.in_degree_centrality(g))
    top_5_outdeg = get_top_five(nx.out_degree_centrality(g))
    top_5_eig = get_top_five(nx.eigenvector_centrality(g))
    top_5_pr = get_top_five(nx.pagerank(g, alpha=0.9))
    hubs, auth = nx.hits(g)
    top_5_hubs = get_top_five(hubs)
    top_5_auth = get_top_five(auth)
    print("top in", top_5_indeg)
    print("top out", top_5_outdeg)
    print("top eig", top_5_eig)
    print("top pr", top_5_pr)
    print("top hubs", top_5_hubs)
    print("top auth", top_5_auth)


def degree_distribution(g):
    # PART 1 A
    matrix = nx.adjacency_matrix(g)
    size = matrix.shape[0]
    # Compute Degree Distribution ------>
    # Compute degree of each node from adjacency matrix; Create dict that will contain {node : degree (k)}
    degreeCount = dict()
    for i in range(size):
        sum = matrix.getrow(i).sum()
        degreeCount[i] = sum

    # Create another dict that will contain {degree(k): Nk}
    countFreq = dict()
    for i in degreeCount.values():
        if i in countFreq:
            countFreq[i] = countFreq[i] + 1
        else:
            countFreq[i] = 1

    # Finally, we want to store {k: pk}
    countProb = dict()
    for (i, j) in countFreq.items():
        countProb[i] = countFreq[i] / size

    # Verify sum (should add up to 1 since probability)
    # sum = 0
    # for y in countProb.values():
    #    sum = sum + y

    # Store degrees (x) and their probabilities (y)
    x = countFreq.keys()
    y = countProb.values()

    # Turn dict components into arrays to use for linear regression
    x_array = np.fromiter(countFreq.keys(), dtype=float)
    y_array = np.fromiter(countProb.values(), dtype=float)

    # Bin the data
    bin_means, bin_edges, binnumber = stats.binned_statistic(x_array, y_array,
                                                             statistic='mean', bins=25)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    # Display distribution on scatterplot
    plt.xlabel("k")
    plt.ylabel("p$k$")
    plt.plot((binnumber - 0.5) * bin_width, y_array, 'b.', alpha=0.5)

    # log log scale
    plt.yscale('log')
    plt.xscale('log')

    # make a best fit line
    # get rid of any 0s to prevent things from getting messed up later on
    min_nonzero_x = np.min(x_array[np.nonzero(x_array)])
    x_array[x_array == 0] = min_nonzero_x
    min_nonzero_y = np.min(y_array[np.nonzero(y_array)])
    y_array[y_array == 0] = min_nonzero_y

    # unlog the x and y array to make a line
    logx = np.log(x_array)
    logy = np.log(y_array)

    # get the slope coefficients
    m, c = np.polyfit(logx, logy, 1)
    print("Slope: ", m)
    y_fit = np.exp(m * np.unique(logx) + c)

    plt.plot(np.unique(x_array), y_fit, 'r')
    plt.show()


def connected_components(g):
    # Part 1 D
    # Compute Number of Connected Components ---->
    matrix = nx.adjacency_matrix(g)
    size = matrix.shape[0]
    matrix_convert = csr_matrix(matrix)
    conn_components, labels = connected_components(csgraph=matrix_convert, directed=False, return_labels=True)

    components = dict()
    # Divide components up and store them as: {Component: # nodes}
    for l in labels:
        if l in components:
            components[l] = components[l] + 1
        else:
            components[l] = 1

    sizeFreq = dict()
    #  {# nodes: frequency}
    for i in components.values():
        if i in sizeFreq:
            sizeFreq[i] = sizeFreq[i] + 1
        else:
            sizeFreq[i] = 1

    x = sizeFreq.keys()
    y = sizeFreq.values()

    x = list(sizeFreq.keys())
    y = list(sizeFreq.values())

    print(sizeFreq)
    # Display distribution on scatterplot
    plt.xlabel("Size of Components (Nodes)")
    plt.ylabel("Frequency")
    plt.title("Size of Connected Components of a Synthetic Network")
    # plt.scatter(x,y)
    plt.xlim(0, 20)
    plt.ylim(0, 50)
    plt.bar(x, y, color='blue', width=1)
    # plt.scatter(x,y)
    plt.show()


if __name__ == '__main__':
    with open('./user_data/biden_user_subred_freq.json') as json_file:
        biden_data = json.load(json_file)
    with open('./user_data/trump_user_subred_freq.json') as json_file:
        trump_data = json.load(json_file)
    users = user_network(biden_data, trump_data, "All_Users")
    subs = global_subreddit_network(biden_data, trump_data, "All")
    degree_distribution(subs)
    #run_classification(users, node_classification.local_and_global_consistency, "Users")
    #some_users = subreddit_network(trump_data, "tump_users")
    #top_5(some_users)
    #some_users = subreddit_network(biden_data, "biden_users")
    #top_5(some_users)

