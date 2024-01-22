import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from jax import random
import jraph
import numpy as np
from gnn_operations import gcn_layer

num_places = 20
num_pressure_levels = 3

latitude = jnp.linspace(-90, 90, num_places)
longitude = jnp.linspace(-180, 180, num_places)

def generate_linear_features(timestep, latitudes, longitudes, pressure_level, key, last_values):
    num_places = len(latitudes)

    base_temperature = 20 + latitudes * 0.3 + longitudes * 0.1 + pressure_level * 0.1 
    base_humidity = latitudes * 0.4 + 50 + pressure_level * 0.2
    base_wind_speed = jnp.abs(longitudes) * 0.1 + pressure_level * 0.1  

    # Noise generation for each feature
    temperature_noise = random.normal(key, shape=(num_places,)) * 2 - 1
    humidity_noise = random.normal(key, shape=(num_places,)) * 3
    wind_speed_noise = random.normal(key, shape=(num_places,)) * 3


    if not last_values.get(pressure_level):
        last_values[pressure_level] = {
            'temperature': base_temperature,
            'humidity': base_humidity,
            'wind_speed': base_wind_speed
        }
    diurnal_temp_variation = 3*np.sin(np.pi * timestep/4)
    # Retrieve the last known values for the current  pressure level
    last_temp = last_values[pressure_level]['temperature']
    last_humid = last_values[pressure_level]['humidity']
    last_wind = last_values[pressure_level]['wind_speed']
    
    # Calculate the new features with vectorized operations
    temperatures = base_temperature * 0.9 + diurnal_temp_variation  + temperature_noise * 0.2
    humidities = base_humidity * 0.9 + humidity_noise 
    wind_speeds = base_wind_speed * 0.9 + wind_speed_noise * 0.2 + diurnal_temp_variation

    # Ensure values are within a valid range
    # sine_fluctuation = jnp.sin(timestep)
    # temperatures += sine_fluctuation
    # humidities += sine_fluctuation
    # wind_speeds += sine_fluctuation

    # temperatures = np.clip(temperatures, -20, 45)
    humidities = np.clip(humidities, 0, 100)
 

    # Stack features into a single array
    features = np.column_stack((temperatures, humidities, wind_speeds))
    
    last_values[pressure_level] = {
        'temperature': temperatures,
        'humidity': humidities,
        'wind_speed': wind_speeds
    }
    
    return features, last_values


def euclidean_distance(point1, point2):
    return jnp.linalg.norm(point1 - point2)

# each pressure level, each timestep
def generate_graph_for_pressure_level(timestep, pressure_level, latitude, longitude, K, key, last_values):
    
    nodes, last_values = generate_linear_features(timestep, latitude, longitude, pressure_level, key, last_values)
    senders = []
    receivers = []
    edge_distances = []
    
    # Iterate over all powers of 2 up to 2^K
    for K in range(0, K + 1):
        
        distance = 2 ** K  # Compute the distance as 2^K
        # print(distance)
        for i in range(num_places):
            # Connect to the `distance`-th neighbor clockwise
            neighbor_clockwise = (i + distance) % num_places
            senders.append(i + pressure_level * num_places)
            receivers.append(neighbor_clockwise + pressure_level * num_places)
            edge_distances.append(euclidean_distance(nodes[i], nodes[neighbor_clockwise]))
            
            # Connect to the `distance`-th neighbor counter-clockwise
            neighbor_counter_clockwise = (i - distance) % num_places
            senders.append(i + pressure_level * num_places)
            receivers.append(neighbor_counter_clockwise + pressure_level * num_places)
            edge_distances.append(euclidean_distance(nodes[i], nodes[neighbor_counter_clockwise]))

    return nodes, senders, receivers, edge_distances, last_values


# connect pressure levels in one timestep's graph
def connect_pressure_levels(all_nodes, senders, receivers, edge_distances):
    for i in range(num_pressure_levels - 1):
        for node in range(num_places):
            sender = node + i * num_places
            receiver = node + (i + 1) * num_places
            distance = euclidean_distance(all_nodes[sender], all_nodes[receiver])

            senders.append(sender)
            receivers.append(receiver)
            edge_distances.append(distance)
            
    return senders, receivers, edge_distances

def concentric_layout(G):
    pos = {}
    for i in range(G.number_of_nodes()):
        angle = 2 * jnp.pi * (i % num_places) / num_places
        radius = 1 + (i // num_places)
        pos[i] = (radius * jnp.cos(angle), radius * jnp.sin(angle))
    return pos


def generate_graph(timestep, K, has_interlevel_connections, key, last_values):
    all_nodes, all_senders, all_receivers, all_edge_distances = [], [], [], []
    

    for pl in range(num_pressure_levels):
        key, subkey = jax.random.split(key)
        nodes, senders, receivers, edge_distances, last_values = generate_graph_for_pressure_level(timestep, pl, latitude, longitude, K, subkey, last_values)

        # nodes, senders, receivers, edge_distances = generate_graph_for_pressure_level(pl, latitude, longitude, K, key)
        all_nodes.extend(nodes)
        all_senders.extend(senders)
        all_receivers.extend(receivers)
        all_edge_distances.extend(edge_distances)

    if has_interlevel_connections:
        all_senders, all_receivers, all_edge_distances = connect_pressure_levels(all_nodes, all_senders, all_receivers, all_edge_distances)

    # Convert the lists to jax.numpy arrays
    all_nodes = jnp.array(all_nodes)
    all_senders = jnp.array(all_senders)
    all_receivers = jnp.array(all_receivers)
    all_edge_distances = jnp.array(all_edge_distances).reshape(-1, 1)  # Reshaping the edge_distances for edge features

    # Construct the GraphsTuple
    graph_tuple = jraph.GraphsTuple(
        n_node=jnp.array([len(all_nodes)]), 
        n_edge=jnp.array([len(all_edge_distances)]), 
        nodes=all_nodes,
        edges=all_edge_distances,  # Using edge distances as edge features
        senders=all_senders,
        receivers=all_receivers,
        globals=None  
    )
    # G = nx.Graph()
    # edges = [(s.item(), r.item()) for s, r in zip(graph_tuple.senders, graph_tuple.receivers)]
    # G.add_edges_from(edges)

    # plt.figure(figsize=(10, 10))
    # pos = concentric_layout(G)
    # nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue')
    # plt.show()

    

    return graph_tuple, last_values
    
target_graph = [0]


def generate_batched_graph_with_K_bool_timesteps(K, bool, number_of_timesteps, keys):
    """ K is the power of 2 that defines the edge connections in the graph. e.g. K=2 means each node connects with neighbors that are 4 nodes away.
    bool is a boolean indicating whether we are connecting between pressure levels. keys defines the randomness we introduces with the temporal component."""
    last_values = {}
    
    graphs = []
    for i in range(number_of_timesteps):
        graph, last_values = generate_graph(i, K, bool, keys[i], last_values)
        graphs.append(graph)
    return graphs


from jax import random, vmap
import os 
import pickle
output_directory = './dataset'

num_timesteps = 10
num_samples_per_K = 25  # Number of graph instances to generate for each K
K_values = [0, 1, 2, 3]
bool_interlevel = True  # Assuming you want inter-level connections
split_ratios = (0.7, 0.15, 0.15)  # Train, validation, test split ratios

# vectorized normalization function
def normalize_graphs_tuple(graphs_tuple, mean_features, std_features):
    normalized_nodes = (graphs_tuple.nodes - mean_features) / std_features
    return graphs_tuple._replace(nodes=normalized_nodes)

vectorized_normalize_graphs_tuple = vmap(normalize_graphs_tuple, in_axes=(0, None, None))

def generate_and_normalize_graphs(K, bool_interlevel, num_timesteps, num_samples_per_K, split_ratios):
    all_samples = []
    for sample_index in range(num_samples_per_K):
        key = random.PRNGKey(sample_index)
        keys = [random.fold_in(key, i) for i in range(num_timesteps)]
        graph_instance = generate_batched_graph_with_K_bool_timesteps(K, bool_interlevel, num_timesteps, keys)
        gcn_graph_instance = [gcn_layer(g) for g in graph_instance]
       
    # Normalize each graph individually
        normalized_graphs = []
        for graph in gcn_graph_instance:
            mean_features = jnp.mean(graph.nodes, axis=0)
            std_features = jnp.std(graph.nodes, axis=0)
            std_features = jnp.where(std_features == 0, 1, std_features)  # Avoid division by zero
            normalized_graph = normalize_graphs_tuple(graph, mean_features, std_features)
            normalized_graphs.append(normalized_graph)
        
        all_samples.append(normalized_graphs)
    
    # Split the list of graph samples into train, validation, and test sets
    train_size = int(split_ratios[0] * num_samples_per_K)
    val_size = int(split_ratios[1] * num_samples_per_K)
    train_graphs = all_samples[:train_size]
    val_graphs = all_samples[train_size:train_size + val_size]
    test_graphs = all_samples[train_size + val_size:]
    print(len(train_graphs))
    print(type(train_graphs[0]))
    return {
        'train': train_graphs,
        'validation': val_graphs,
        'test': test_graphs
    }

def save_datasets(graph_datasets, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for K, datasets in graph_datasets.items():
        for split, graphs in datasets.items():
            filename = f'graphs_K{K}_{split}.pkl'
            with open(os.path.join(output_dir, filename), 'wb') as f:
                pickle.dump(graphs, f)

def main(output_dir, num_timesteps, num_samples_per_K, K_values, bool_interlevel, split_ratios):
    graph_datasets = {}
    for K in K_values:
        graph_datasets[K] = generate_and_normalize_graphs(K, bool_interlevel, num_timesteps, num_samples_per_K, split_ratios)
    
    # Save datasets to the specified output directory
    save_datasets(graph_datasets, output_dir)
    print(f"Datasets generated and saved to {output_dir}")

if __name__ == "__main__":
    main()

