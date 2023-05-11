import numpy as np

def rwr(start_node, adj_matrix, restart_prob, num_steps):
    

    curr_node = start_node
    seq=[start_node]
    
    for i in range(num_steps):
        # with probability restart_prob, reset the probabilities to the initial values
        if np.random.rand() < restart_prob:
            curr_node = start_node
            seq.append(curr_node)
        else:
            # otherwise, take a step in the random walk
            neighbours = adj_matrix[curr_node]            
            neighbours = neighbours/np.sum(neighbours)

            curr_node = np.random.choice(len(neighbours), p=neighbours)
            seq.append(curr_node)
    
    return seq


def get_ego_net(seq, centre_node):
    """Returns the ego network of a given node in a random walk sequence."""
    
    ego_net = []

    for i in range(len(seq)):
        if seq[i] == centre_node:
            if i<len(seq)-1 and seq[i+1] not in ego_net and seq[i+1]!=centre_node:
                ego_net.append(seq[i+1])
    
    return ego_net


adj_matrix = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [1, 1, 1, 0]])

seq = rwr(0, adj_matrix, 0.2, 8)
print(seq)

print(get_ego_net(seq, 0))