import torch

def fc_edge_index(n_nodes: int) -> torch.Tensor:
    assert isinstance(n_nodes, int)
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive int")
    n_edges = n_nodes * (n_nodes - 1)
    edge_index = torch.empty((2, n_edges), dtype=torch.long)
    c = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_index[0][c] = i
                edge_index[1][c] = j
                c += 1
    return edge_index