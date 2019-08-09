import networkx as nx
import numpy as np
import pickle


class TimeGraph:
    def __init__(self, edgelist, time_prop_name):
        self.G = nx.MultiDiGraph()
        self.G.add_edges_from(edgelist)
        self.time_prop_name = time_prop_name#'time'
        self.edges = list(self.G.edges(data=True))

        self.beta = None # number of temporal context windows
        self.D = None # embedding dimension 128
        self.L = None # max walk length
        self.omega = None # min walk length / context window size for skip gram

        self.all_time_walks = None
        self.num_nodes = len(list(self.G.nodes()))
        self.num_edges = len(self.edges)

    def set_temporal_walk_params(self, beta, D, omega, L=80):
        self.beta = beta
        self.D = D
        self.L = L
        self.omega = omega

    def sample_edge(self):
        num_edges = len(self.edges)
        choice = np.random.choice(num_edges)
        return self.edges[choice]

    def generate_ctdne_walks(self): # Algorithm 1 from CTDNE paper
        # collection of random walks
        all_time_walks = []

        # initialize number of context windows
        C = 0
        counter = 0
        while self.beta - C > 0:
            u, v, prop = self.sample_edge()
            t = prop[self.time_prop_name]

            walk_t = self._temporal_walk(start_edge=(u, v), t=t, C=self.omega + self.beta - C - 1)
            if (walk_t is not None) and (len(walk_t) >= self.omega):
                all_time_walks.append(walk_t)
                C = C + (len(walk_t) - self.omega + 1)

            counter += 1
            if (counter + 1) % 1000 == 0:
                print('Loop ran for ', counter + 1, 'times!\t Current C = ', C)

        self.all_time_walks = all_time_walks

    def _temporal_walk(self, start_edge=None, t=None, C=0): # Algorithm 2 from CTDNE
        """ Returns a random walk that respects time
            start: the start node of the random walk.
        """
        G = self.G
        prop_name = self.time_prop_name

        if start_edge:
            path = [start_edge[0], start_edge[1]]
        else:
            raise ValueError('start_edge should not be None.')

        curr_node = start_edge[1]

        for p in range(1, min(self.L, C)):
            # select next nodes whose time (curr_node, next_node) is greater than t
            neighbor_candidates = []
            for u, v, prop in G.out_edges(curr_node, data=True):
                if prop[prop_name] >= t:
                    neighbor_candidates.append((v, prop[prop_name]))

            # check if there are valid neighbors to walk to
            if len(neighbor_candidates) > 0:
                # set new current node & t
                idx_next_node = np.random.choice(range(len(neighbor_candidates)))
                curr_node, t = neighbor_candidates[idx_next_node]

                # add new current node to path
                path.append(curr_node)
            else:
                break
        return path


def main():

    path = '../../network_data/ia-contact/ia-contact.time.walks'
    save_path = '../../network_data/ia-contact/'
    with open(save_path + 'embedding_edges', 'rb') as f:
        embedding_edges = pickle.load(f)

    timeG =TimeGraph(embedding_edges, 'time')

    R = 80
    N = timeG.num_nodes
    omega = 10
    L = 80
    beta = R * N * (L - omega + 1)

    print("Beta value:", beta)

    print("Started Walk...")

    timeG.set_temporal_walk_params(beta=beta, D=128, omega=omega, L=L)
    timeG.generate_ctdne_walks()

    ctdne_walks = timeG.all_time_walks

    with open(path, 'wb') as f:
        pickle.dump(ctdne_walks, f)


if __name__ == '__main__':
    main()
