import pickle, os
import jax.numpy as jnp
import numpy as np
from training import loss_fn, loss_truth_prediction, stack_graphs_and_targets

def test(model_params, test_input_graphs, test_target_graphs):
    """
    Test the GNN model on the provided sequence of test data.

    :param model_params: The parameters of the trained model.
    :param test_input_graphs: The sequence of input graphs for testing.
    :param test_target_graphs: The sequence of target graphs for testing.
    :param k: The value of k used during training.
    :return: The average loss computed on the test data.
    """
    losses = []
    for input_graph, target_graph in zip(test_input_graphs, test_target_graphs):
        # Use the same loss function from training for testing
        loss = loss_fn(model_params, input_graph, target_graph.nodes)
        losses.append(loss)
    # Return the average loss over all the graph pairs
    #print(losses)
    return jnp.mean(jnp.array(losses))
    



test_data_dir = './dataset' 
K_values = [0, 1, 2, 3]  


def load_test_data(K_values, test_data_dir):
    test_data = {}
    for Kd in K_values:
        test_data_path = os.path.join(test_data_dir, f'graphs_K{Kd}_test.pkl')
        with open(test_data_path, 'rb') as f:
            test_data[Kd] = pickle.load(f)
    return test_data

test_data = load_test_data(K_values, test_data_dir)
num_samples = len(test_data[0])  # You need to load the number of samples from your test data

prediction_results = {Kd: {sample_id: {Km: None for Km in K_values} 
                            for sample_id in range(num_samples)} 
                      for Kd in K_values}

ground_truths = {Kd: {sample_id: {Km: None for Km in K_values} 
                       for sample_id in range(num_samples)} 
                 for Kd in K_values}


def run_testing_for_all_samples(trained_models, test_data_dir, K_values, stack_size):
    test_data = load_test_data(K_values, test_data_dir)
    test_results = {Kd: {Km: [] for Km in K_values} for Kd in K_values}

    # Iterate over each Kd in the test datasets
    for Kd in K_values:
        # Iterate over each sample in the Kd's test dataset
        for sample_id, sample in enumerate(test_data[Kd]):
            test_input_graphs, test_target_graphs = stack_graphs_and_targets(sample, stack_size)

            # Iterate over each Km model trained on this Kd dataset
            for Km in K_values:
                model_params = trained_models[Kd][Km][0]
                
                
                average_loss, ground_truth, predictions = loss_truth_prediction(model_params, test_input_graphs, test_target_graphs)
                
                prediction_results[Kd][sample_id][Km] = predictions
                ground_truths[Kd][sample_id][Km] = ground_truth
                test_results[Kd][Km].append(average_loss)

    # Calculate the average loss across all samples for each Kd and Km
    average_test_results = {
        Kd: {Km: np.mean(losses) for Km, losses in Km_results.items()}
        for Kd, Km_results in test_results.items()
    }

    return average_test_results, prediction_results, ground_truths

def main():
    test_data_dir = './dataset' 
    K_values = [0, 1, 2, 3]  # Define or load your K_values
    stack_size = 2  # Define or load your stack_size

    # Load trained models (this assumes they are saved in a file named 'trained_models.pkl')
    with open('trained_models.pkl', 'rb') as f:
        trained_models = pickle.load(f)

    # Call the testing function
    test_loss_results, prediction_results, ground_truths = run_testing_for_all_samples(trained_models, test_data_dir, K_values, stack_size)

    # Save the results to a file or process them as needed
    with open('test_results.pkl', 'wb') as f:
        pickle.dump((test_loss_results, prediction_results, ground_truths), f)

    # Print out the average test loss for each Kd, Km combination
    for Kd in K_values:
        for Km in K_values:
            print(f"Average loss for Data K={Kd}, Model K={Km}: {test_loss_results[Kd][Km]}")

if __name__ == "__main__":
    main()