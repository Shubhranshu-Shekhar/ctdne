import networkx as nx
import numpy as np
import pickle


def load_data_as_graph(path='network_data/ia-contact.edges', weight_idx=2, time_idx=3):
    '''
    Returns a networkx graph.
    Edge property is called 'time'
    :param path: path the to dataset with header u, v, weight(u, v), time(u, v)
    :return: G
    '''
    edges = []
    with open(path) as f:
        for line in f:
            tokens = line.strip().split()
            u = int(tokens[0])
            v = int(tokens[1])
            time = int(tokens[time_idx])
            if weight_idx:
                weight = int(tokens[weight_idx])

                # add edge
                edges.append((u, v, {'weight': weight, 'time':time}))
            else:
                edges.append((u, v, {'time': time}))

    g = nx.MultiDiGraph()
    g.add_edges_from(edges)
    print(g.edges(2, data=True))

    return g


def get_negative_edge(g, first_node=None):
    if first_node is None:
        first_node = np.random.choice(g.nodes())  # pick a random node
    possible_nodes = set(g.nodes())
    neighbours = [n for n in g.neighbors(first_node)] + [first_node]
    possible_nodes.difference_update(neighbours)  # remove the first node and all its neighbours from the candidates
    second_node = np.random.choice(list(possible_nodes))  # pick second node
    edge = (first_node, second_node, {'weight':1, 'time': None})
    return edge


def create_embedding_and_training_data_old(g, train_edges_fraction=0.75):
    edges = sorted(g.edges(data=True), key=lambda x: x[2]['time'])
    num_edges = len(edges)

    # training edges
    num_train_edges = int(train_edges_fraction * num_edges)
    train_edges = edges[:num_train_edges]

    # link prediction positive edges
    pos_edges = edges[num_train_edges:]
    neg_edges = []
    for i in range(len(pos_edges)):
        n_edge = get_negative_edge(g)
        neg_edges.append(n_edge)

    return train_edges, pos_edges, neg_edges


def create_embedding_and_training_data(g, train_edges_fraction=0.75):
    '''
    Create partition of edges into
     -- embedding edges used for learning the embedding
     -- pos edges : positive example of edges for link prediction task

    :param g: nx graph
    :param train_edges_fraction: what fraction of edges to use for embedding learning
    '''

    nodes = g.nodes()
    train_edges = []
    pos_edges = []
    neg_edges = []

    for node in nodes:
        edges_of_node = []
        for e in g.edges(node, data=True): # only gets outgoing edges
            edges_of_node.append(e)

        edges_of_node = sorted(edges_of_node, key=lambda x: x[2]['time'])
        num_edges = len(edges_of_node)

        # training edges per node
        num_train_edges = int(train_edges_fraction * num_edges)
        train_edges.extend(edges_of_node[:num_train_edges])

        # link prediction positive edges
        pos_edges.extend(edges_of_node[num_train_edges:])

    for i in range(len(pos_edges)):
        n_edge = get_negative_edge(g)
        neg_edges.append(n_edge)

    return train_edges, pos_edges, neg_edges


def main():
    path = '../../network_data/ia-contact/ia-contact.edges'
    contact_g =  load_data_as_graph(path=path, weight_idx=2, time_idx=3)
    embedding_edges, pos_edges, neg_edges = create_embedding_and_training_data(contact_g, train_edges_fraction=0.75)

    save_path = '../../network_data/ia-contact/'
    with open(save_path + 'embedding_edges', 'wb') as f:
        pickle.dump(embedding_edges, f)
    with open(save_path + 'pos_edges', 'wb') as f:
        pickle.dump(pos_edges, f)
    with open(save_path + 'neg_edges', 'wb') as f:
        pickle.dump(neg_edges, f)

if __name__ == '__main__':
    main()
