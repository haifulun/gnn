import jraph
import jax.numpy as jnp

def aggregate_fn(sent_attributes, received_attributes, global_attributes):
    """Aggregate function for GNN which takes the mean of received attributes."""
    return jnp.mean(received_attributes, axis=0)

def update_node_fn(node_features, aggregated_messages):
    """Update node features by combining with aggregated messages; here we simply take the mean."""
    return (node_features + aggregated_messages) / 2

def gcn_layer(graph):
    """Defines a single layer of a Graph Convolutional Network."""
    return jraph.GraphsTuple(
        n_node=graph.n_node,
        n_edge=graph.n_edge,
        nodes=update_node_fn(graph.nodes, aggregate_fn(
            sent_attributes=graph.nodes[graph.senders],  # Messages are the features of the sender nodes.
            received_attributes=graph.nodes[graph.receivers],  # Received node attributes.
            global_attributes=None  # No global attributes used here.
        )),
        edges=graph.edges,
        globals=graph.globals,
        senders=graph.senders,
        receivers=graph.receivers,
    )

def apply_gcn_layers(graph, num_layers):
    """Applies a number of GCN layers to a graph."""
    for _ in range(num_layers):
        graph = gcn_layer(graph)
    return graph