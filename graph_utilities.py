import jax.numpy as jnp
import jraph
def euclidean_distance(point1, point2):
    return jnp.linalg.norm(point1 - point2)

def concentric_layout(G, num_places, num_pressure_levels):
    pos = {}
    for i in range(G.number_of_nodes()):
        angle = 2 * jnp.pi * (i % num_places) / num_places
        radius = 1 + (i // num_pressure_levels)
        pos[i] = (radius * jnp.cos(angle), radius * jnp.sin(angle))
    return pos

def stack_graphs_and_targets(graphs, stack_size):
   
    stacked_graphs = []
    target_graphs = []

    

    for i in range(len(graphs) - stack_size):
        # Extract a subset of graphs to stack
        subset_graphs = graphs[i:i + stack_size]

        # Concatenate node features across timesteps for each graph in the subset
        concatenated_nodes = jnp.concatenate([g.nodes for g in subset_graphs], axis=1)

        # Create a new GraphsTuple with concatenated features
        stacked_graph = jraph.GraphsTuple(
            nodes=concatenated_nodes,
            edges=subset_graphs[0].edges,
            senders=subset_graphs[0].senders,
            receivers=subset_graphs[0].receivers,
            globals=subset_graphs[0].globals,
            n_node=subset_graphs[0].n_node,
            n_edge=subset_graphs[0].n_edge,
        )
        stacked_graphs.append(stacked_graph)

        # The target graph is the next graph in the sequence
        target_graph = graphs[i + stack_size]
        target_graphs.append(target_graph)

    return stacked_graphs, target_graphs
