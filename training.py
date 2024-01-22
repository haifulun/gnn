import jraph
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle, os
from data_generation import connect_pressure_levels, generate_graph_for_pressure_level
from graph_utilities import stack_graphs_and_targets
import random


num_places = 20
num_pressure_levels = 3

latitude = jnp.linspace(-90, 90, num_places)
longitude = jnp.linspace(-180, 180, num_places)

# Define the GNN model

def mlp_network(output_size):
    """Defines an MLP network for node and edge updates."""
    mlp = hk.Sequential([
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(output_size)
    ])
    return mlp


def aggregate_fn(sent_attributes, receiver_indices, num_segments):
    # Use `jraph.segment_mean` to compute the mean of received attributes for each node.
    #print(sent_attributes.shape, receiver_indices.shape)
    return jraph.segment_mean(sent_attributes, receiver_indices, num_segments)
    # graph.nodes[graph.senders],  # Messages are the features of the sender nodes.
    #             graph.receivers,  # Index into the receiver nodes.
    #             num_segments=graph.nodes.shape[0]  # The number of nodes.

class GNN(hk.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.node_mlp = mlp_network(self.output_size)  # MLP for node updates

    def __call__(self, graph):
        # Node update function using an MLP
        def update_node_fn(nodes, sent_attributes, received_attributes, global_attributes):
            # Process the aggregated messages with the MLP network
            # `received_attributes` are already aggregated by `GraphNetwork`!!
            return self.node_mlp(received_attributes)
        
        # Create a GraphNetwork with the specified update functions
        graph_network = jraph.GraphNetwork(
            update_node_fn=update_node_fn,
            update_edge_fn=None,  # No edge updates?   edges are fixed distances
            aggregate_edges_for_nodes_fn=jraph.segment_mean,  # Aggregation function for message passing
            update_global_fn=None  # Assuming no global updates for simplicity
        )

        # Apply the GraphNetwork to update the graph
        return graph_network(graph)



class MultiLayerGNN(hk.Module):
    def __init__(self, output_size, num_layers, name=None):
        super().__init__(name=name)
        self.layers = [GNN(output_size) for _ in range(num_layers - 1)]
        self.final_layer = GNN(output_size)  # The final layer (presumably linear)

    def __call__(self, graph):
        # Process the graph through each GNN layer in sequence except the last one
        for layer in self.layers[:-1]:  # Exclude the final layer from this loop
            graph = layer(graph)
            # Apply an activation function between the GNN layers
            graph = graph._replace(nodes=jax.nn.relu(graph.nodes))

        # For the final layer, apply without an additional activation
        graph = self.final_layer(graph)  # Assumes final layer output is linear
        
        return graph


def model_fn(graph):
    model = MultiLayerGNN(output_size=3, num_layers=3)
    return model(graph)

# Transform the model function into a Haiku model
model = hk.without_apply_rng(hk.transform(model_fn))


def single_sample_loss_fn(params, input_graph, target_graph):
    prediction = model.apply(params, input_graph).nodes
    return jnp.mean(jnp.square(prediction - target_graph.nodes))

def loss_fn(params, input_graphs, target_graphs):
    loss = 0
    predictions = []
    ground_truth = []
    for input_graph, target in zip(input_graphs, target_graphs):
       
        prediction = model.apply(params, input_graph).nodes
        
        loss += jnp.mean(jnp.square(prediction - target.nodes))
   
    return jnp.sqrt(loss / (len(target_graphs)))
# Vectorize the single_sample_loss_fn to handle batches


# batch_loss_fn = jax.vmap(single_sample_loss_fn, in_axes=(None, 0, 0))
# def loss_fn(params, batch_input_graphs, batch_target_graphs):
   
    
#     # Apply the batched loss function
#     losses = batch_loss_fn(params, batch_input_graphs, batch_target_graphs)
    
#     # Return the average loss across the batch
#     return jnp.mean(losses)
# cannot vectorize n_nodes, n_edge dimension not right ??????

def loss_truth_prediction(params, input_graphs, target_graphs):
    
    loss = 0
    predictions = []
    ground_truth = []
    for input_graph, target in zip(input_graphs, target_graphs):
       
        prediction = model.apply(params, input_graph).nodes
        ground_truth.append(target.nodes)
        predictions.append(prediction)

        
        loss += jnp.mean(jnp.square(prediction - target.nodes))
        
    return loss / (len(target_graphs)), ground_truth, predictions

# Training function
def train(input_graphs, target_graphs, val_input_graphs, val_target_graphs, learning_rate=0.001, num_epochs=40, k=3, patience=10):
    # Initialize model parameters
    #params = hk.transform(lambda g: GNN(output_size=target_graphs[0].nodes.shape[-1], k=k)(g)).init(jax.random.PRNGKey(42), input_graphs[0])
    # model = hk.transform(lambda g: MultiLayerGNN(output_size=3, num_layers=k)(g))
    print(input_graphs[0].nodes.shape)
    params = model.init(jax.random.PRNGKey(42), input_graphs[0])
    preds = []
    # Initialize optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_params = None
    # Training loo3
    training_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        epoch_losses_train = []
        epoch_losses_val = []
   
        
        grads = jax.grad(loss_fn)(params, input_graphs, target_graphs) # use target.nodes as the prediction is for nodes
       

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
            
        # Compute training loss
        train_loss = loss_fn(params, input_graphs, target_graphs)
        # epoch_losses_train.append(train_loss)
            # losses.append(loss)
        print(f"avergate training loss this epoch: {train_loss}")
        training_losses.append(train_loss)
      
        grads = jax.grad(loss_fn)(params, val_input_graphs, val_target_graphs)
            
            
        val_loss = loss_fn(params, val_input_graphs, val_target_graphs)
       
        validation_losses.append(val_loss)

    # Early stopping check
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
         
            best_params = params.copy() 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch}")
                break
    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    # Use the best_params for the final model parameters if early stopping was triggered
    final_params = best_params if best_params is not None else params

    return final_params, 0






def load_datasets_from_pkl(K_values, dataset_dir):
    graph_datasets = defaultdict(dict)
    for K in K_values:
        for split in ['train', 'validation']:
            file_path = os.path.join(dataset_dir, f'graphs_K{K}_{split}.pkl')
            with open(file_path, 'rb') as file:
                graph_datasets[K][split] = pickle.load(file)
    return graph_datasets

def generate_edges_for_km_connectivity(nodes, Km, num_places):
    senders = []
    receivers = []
    edge_distances = []
    
    for pressure_level in range(num_pressure_levels):
        # Use the provided edge generation logic for the current pressure level
        # This might involve calling a function similar to generate_graph_for_pressure_level
        # but with the new Km value instead of the old K
        key = random.PRNGKey(42) # this doesn't matter as we are not using node features
        _, new_senders, new_receivers, new_edge_distances, _ = generate_graph_for_pressure_level(
            1, pressure_level, latitude, longitude, Km, key, {}
        )
        senders.extend(new_senders)
        receivers.extend(new_receivers)
        edge_distances.extend(new_edge_distances)

    return senders, receivers, edge_distances
    
def modify_connectivity_for_sample(sample, Km, num_places):
    modified_sample = []
    #print(len(sample[0]))
    for i in range(len(sample)):
        timestep_graph = sample[i]
        
        nodes = timestep_graph.nodes
        # Generate new edges based on Km connectivity
        senders, receivers, edge_distances = generate_edges_for_km_connectivity(nodes, Km, num_places)
        # Create a new graph with the same nodes but updated edges
       
        senders, receivers, edge_distances = connect_pressure_levels(nodes, senders, receivers, edge_distances)

        # Convert the lists to jax.numpy arrays
        all_nodes = jnp.array(nodes)
        all_senders = jnp.array(senders)
        all_receivers = jnp.array(receivers)
        all_edge_distances = jnp.array(edge_distances).reshape(-1, 1)  # Reshaping the edge_distances for edge features

        # Construct the GraphsTuple
        modified_graph = jraph.GraphsTuple(
            n_node=jnp.array([len(all_nodes)]), 
            n_edge=jnp.array([len(all_edge_distances)]), 
            nodes=all_nodes,
            edges=all_edge_distances,  # Using edge distances as edge features
            senders=all_senders,
            receivers=all_receivers,
            globals=None  
        )
        modified_sample.append(modified_graph)
    # print(len(modified_sample))
    return modified_sample

def rollout_and_train(graph_datasets, K_values, num_places, stack_size):
    trained_models = {Kd: {Km: None for Km in K_values} for Kd in K_values}

    for Kd in K_values:
        for Km in K_values:
            all_stacked_graphs, all_target_graphs = [], []
            all_val_graphs, all_val_targets = [], []

            for split in ['train', 'validation']:
                dataset = graph_datasets[Kd][split]
                for sample in dataset:
                    # print(len(sample)) == 7
                    # print(type(sample[0]))
                    modified_sample = modify_connectivity_for_sample(sample, Km, num_places)
                    
                    stacked_graphs, target_graphs = stack_graphs_and_targets(modified_sample, stack_size)

                    if split == 'train':
                        all_stacked_graphs.extend(stacked_graphs)
                        all_target_graphs.extend(target_graphs)
                    else:  # validation split
                        all_val_graphs.extend(stacked_graphs)
                        all_val_targets.extend(target_graphs)

            # Training and validation
            trained_params = train(
                all_stacked_graphs, all_target_graphs,
                all_val_graphs, all_val_targets,
                num_epochs=50, patience=10
            )

            trained_models[Kd][Km] = trained_params
            print(f"Data K = {Kd}, Model K = {Km}")

    
    return trained_models
# dataset_dir = './dataset'
# graph_datasets = load_datasets_from_pkl(K_values, dataset_dir)
# stack_size = 2
# trained_models = rollout_and_train(graph_datasets, K_values, num_places, stack_size)


# Now call the function to train all models for each sample and connectivity combination



def main():
    dataset_dir = './dataset'
    K_values = [0, 1, 2, 3]  # Assuming these are the values you want to use
    stack_size = 2  # Or any other value that's appropriate

    # Load the datasets from the pickle files
    graph_datasets = load_datasets_from_pkl(K_values, dataset_dir)

    # Roll out the training process and train models
    trained_models = rollout_and_train(graph_datasets, K_values, num_places, stack_size)

    # Save the trained models to disk
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)

    print("Training complete. Models saved.")

if __name__ == "__main__":
    main()
