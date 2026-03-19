
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import requests
from pprint import pprint
from io import StringIO, BytesIO
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.resource_management import LoadDataset
from uncertainty_engine.nodes.resource_management import Save
from uncertainty_engine.nodes.workflow import Workflow
from uncertainty_engine.graph import Graph
from uncertainty_engine.nodes.resource_management import LoadDataset
from uncertainty_engine.nodes.resource_management import LoadModel
from uncertainty_engine.nodes.resource_management import Download
from uncertainty_engine.nodes.base import Node


# Colours for plotting
KEY_LIME = "#EBF38B"
INDIGO = "#16425B"
INDIGO_50 = "#8AA0AD"
KEPPEL = "#16D5C2"
KEPPEL_50 = "#8AEAE1"
BLACK = "#000000"
GREY_80 = "#333333"
LIGHT_GREY = "#CCCCCC"
WHITE = "#FFFFFF"


###############################################################
# Functions for simulation and plotting of training data and UQ
###############################################################

####
1.
####

def simulate(x_values, noise=True):

    """

        Underlying relationship between flexural strength and neutron irradiation dose.
        'Perfect' relationship can be obtained by prescribing noise=False.

    """

    # baseline flexural strength (MPa)
    sigma0 = 1000

    # Noise addition
    noise_level = 5

    # Model components
    hardening = 0.25 * x_values * np.exp(-0.5 * x_values)      # early radiation hardening
    low_dpa_curvature = 0.3 * x_values * np.exp(-x_values)     # slight positive concavity
    embrittlement = - 0.4 * (1 - np.exp(-0.7 * x_values))       # long-term degradation

    # Synthetic strength model
    y_values = sigma0 * (1 + low_dpa_curvature + hardening + embrittlement)

    if noise:
        # Add noise to the model output
        try:
            y_values += np.random.normal(0, noise_level, size=x_values.shape)

        except:
            y_values += np.random.normal(0, noise_level)

    # Alternativbe ground truth with more complex feature
    # Ensure that plot bounds are updated for the plot_basic and plot_uncertainty functions

    # noise_level = 0.1

    # y_values = 5 + 0.75*np.sin(2*x_values) + 2*np.sin(0.5*x_values)

    # if noise:
    #     try: 
    #         y_values += np.random.normal(0, noise_level, size=x_values)
    #     except:
    #         y_values += np.random.normal(0, noise_level)


    return y_values

####
2.
####

def plot_basic(training_dpa, training_fs):

    """

        Simply plot the ground truth, as a line, and our original samples, as a scatter plot.

    """

    # Irradiation dose (DPA) input space
    dpa = np.linspace(0, 5.5, 100)
    # The ground truth is the model without noise by definition
    ground_truth = simulate(dpa, noise=False)

    y_bounds = [550, 1250]
    # For simulation with more complex features
    # y_bounds = [3, 9]

    plt.figure(figsize=(7,5))
    plt.scatter(training_dpa, training_fs, label="Samples", marker='x', color=INDIGO)
    plt.plot(dpa, ground_truth, linewidth=2, label="Ground Truth", alpha=0.7, color=GREY_80, linestyle='--')
    plt.xlabel("Neutron Irradiation Dose (dpa)")
    plt.ylabel("Flexural Strength (MPa)")
    plt.xlim([-0.5, 5.5])
    plt.ylim(y_bounds)
    plt.legend()
    plt.grid(True)
    plt.savefig("basic_plot.png")

    plt.show()

####
3.
####

def plot_model_uncertainty(training_dpa, training_fs, visualise_means, visualise_stds, visualise_inputs, iteration: int, with_ground_truth=True):

    """

        Plot the posterior mean and shaded plots of 67% and 99% credible intervals.
        Overlay ground truth (optional) and samples at our

    """

    # Ensure that the data is the right shape

    training_dpa=training_dpa.to_numpy().flatten()
    training_fs=training_fs.to_numpy().flatten()
    visualise_means=visualise_means.to_numpy().flatten()
    visualise_stds=visualise_stds.to_numpy().flatten()
    visualise_inputs=visualise_inputs.to_numpy().flatten()

    # Irradiation dose (DPA) input space
    dpa = np.linspace(0, 5.5, 100)
    # The ground truth is the model without noise by definition
    ground_truth = simulate(dpa, noise=False)

    # # Create input space, based on the size of the output
    # input_space = np.linspace(0, 5, len(visualise_means))

    # Confidence interval std multipliers
    z67 = 1.0
    z99 = 2.576

    lower67 = visualise_means - z67 * visualise_stds
    upper67 = visualise_means + z67 * visualise_stds

    lower99 = visualise_means - z99 * visualise_stds
    upper99 = visualise_means + z99 * visualise_stds

    plt.figure(figsize=(7,5))

    y_bounds = [550, 1250]
    # for simulation woth more complex features
    # y_bounds = [3, 9]

    plt.fill_between(visualise_inputs, lower99, upper99, label="99% credible", color=KEPPEL_50)
    plt.fill_between(visualise_inputs, lower67, upper67, label="67% credible", color=KEPPEL)

    plt.plot(visualise_inputs, visualise_means, label='Predicted Mean', color=INDIGO)

    # Only plot the ground truth if selected (selected by default)
    if with_ground_truth:
        plt.plot(dpa, ground_truth, linewidth=2, label="Ground Truth", color=GREY_80, linestyle='--', alpha=0.5)

    plt.scatter(training_dpa, training_fs, label="Samples", marker='x', color=INDIGO)
    plt.legend()
    plt.grid()
    plt.xlabel("Neutron Irradiation Dose (dpa)")
    plt.ylabel("Flexural Strength (MPa)")
    plt.xlim([-0.5, 5.5])
    plt.ylim(y_bounds)

    if with_ground_truth:

        fig_name = "uncertainty_plot_" + str(iteration) + "_gt_overlayed"

    else:

        fig_name = "uncertainty_plot_" + str(iteration)

    plt.savefig(fig_name + ".png")
    plt.show()


#########################
# Engine Specific Helpers
#########################

####
1.
####

def get_dataset(client, resource_name, project_name):

    """

        With only the resource name, returns the content of a dataset
        resource as a pandas DataFrame
    
    
    """

    dataset = client.resources.download(
              resource_id=client.resources.get_resource_id_by_name(
                  name=resource_name,
                  resource_type="dataset",
                  project_id=client.projects.get_project_id_by_name(project_name),
                  ),
              project_id=client.projects.get_project_id_by_name(project_name),
              resource_type="dataset",
      )

    return pd.read_csv(BytesIO(dataset))

####
2.
####

def add_new_sample(client, new_dpa, new_fs, iteration: int, project_name):

    """

        Add a new sample to existing list of samples acquired thus far.


    """

    previous_iteration = iteration - 1

    X_train = get_dataset(
                        client=client
                        , resource_name="x_train_" + str(previous_iteration)
                        , project_name=project_name
                    )
    y_train = get_dataset(
                        client=client
                        , resource_name="y_train_" + str(previous_iteration)
                        , project_name=project_name
                    )


    # Add the new sample taken
    X_train.loc[len(X_train)] = new_dpa
    y_train.loc[len(y_train)] = new_fs

    X_train_csv = X_train.to_csv(index=False)
    y_train_csv = y_train.to_csv(index=False)


    # Save CSV strings to files (overwriting if they exist)
    for filename, data in zip(
        ["x_train.csv", "y_train.csv"]
        , [X_train_csv, y_train_csv]
    ):
        with open(filename, "w") as f:
            f.write(data)

    project_id = client.projects.get_project_id_by_name(project_name)

    # Define an empty dictionary to hold resource IDs (for convenience)
    resource_ids: dict[str,str] = dict()
    resource_names = ["x_train_" + str(iteration), "y_train_" + str(iteration)]

    # Delete existing resources with the same names to avoid conflicts (optional, but useful during development)
    for resource_name in resource_names:
        try:
            existing_resource_id = client.resources.get_resource_id_by_name(

                                        resource_type="dataset"
                                        , project_id=project_id
                                        , name=resource_name

                                    )

            client.resources.delete_resource(

                project_id=project_id
                , resource_type="dataset"
                , resource_id=existing_resource_id

            )
            
            print(f'Deleted existing resource "{resource_name}" with ID {existing_resource_id}')
        except:
            print(f'No existing resource named "{resource_name}" found. Proceeding to upload.')

    # Upload the datasets and store their resource IDs
    for resource_name, file_path in zip(

        resource_names
        , ["x_train.csv", "y_train.csv"]

    ):
        resource_id = client.resources.upload(

            # For resources to be uploaded, they must belong to a project
            project_id=project_id
            , name=resource_name
            , file_path=file_path
            , resource_type="dataset"

        )

        resource_ids[resource_name] = resource_id
        print(f'Uploaded {file_path} as resource "{resource_name}"')

####
3.
####

def train_model(client, project_id, iteration):

    """

        Train a GP on samples avaialabkle in iteration.
    
    """


    # The graph is the backbone of our "TRAIN' workflow - essentially stitching together each of the nodes that will be created
    train_graph = Graph()

    # Get the relevant resource IDs
    x_train_id = client.resources.get_resource_id_by_name(

                    resource_type="dataset"
                    , project_id=project_id
                    , name=f"x_train_{iteration}"

                )
    y_train_id = client.resources.get_resource_id_by_name(

                    resource_type="dataset"
                    , project_id=project_id
                    , name=f"y_train_{iteration}"

                )


    # Creation of the model configuration node
    # Model config is kepy to the default regressor here
    model_config = Node(
        
        node_name="ModelConfig"
        , label="Model Config"
        , client=client

    )

    # Create a LoadDataset node for the x training data
    X_train = LoadDataset(
        
        label="Load Train X"      # the node"s label (needs to be unique in the graph)
        , project_id = project_id # the project id
        , file_id = x_train_id    # the resource id of the file you want to load
        , client=client

    )

    # Create a LoadDataset node for the y training data
    y_train = LoadDataset(
        
        label="Load Train Y"                
        , project_id = project_id                      
        , file_id = y_train_id
        , client=client

    )

    # Add each of the nodes created to the graph
    train_graph.add_node(model_config)
    train_graph.add_node(X_train)
    train_graph.add_node(y_train)

    # Create handles for the configuration and loaded dataset files
    # This essentially allows node outputs to be mapped to the inputs of subsequent nodes
    output_config = model_config.make_handle("config")
    x_handle = X_train.make_handle("file")
    y_handle = y_train.make_handle("file")

    # Create the TrainModel node
    train_model = Node(
        
        node_name="TrainModel"
        , label="Train Model"
        , config=output_config
        , inputs=x_handle
        , outputs=y_handle
        , client=client

    )

    # Name model according to parameters selected in the current iteration
    model_name = f"fs_regressor_{iteration}"

    # Create a node to save the model once trained (in order to make inferences in the future workflows)
    # When thee workflow is created containing this node; the trained model is then saved as resource
    save_model = Save(
        
        label="Save Model"
        , data=train_model.make_handle("model")
        , project_id=project_id
        , file_name=model_name
        , client=client

    )

    # Add the new nodes to the graph
    train_graph.add_node(train_model)
    train_graph.add_node(save_model)

    # The graph is wrapped in a workflow as follows. It is the workflow that can be executed
    train_workflow = Workflow(

        graph=train_graph.nodes
        , inputs=train_graph.external_input
        , external_input_id=train_graph.external_input_id
        , client=client

    )

    # This will train the model, and save the trained model to the project workspace (for subsequent inference)
    train_response = client.run_node(train_workflow)
    assert train_response.status.value == "completed" 

    print("Training complete of " + str(model_name))

    return model_name

####
4.
####

def create_visualise_dataset(client, project_id, input_space_name, iteration):

    """
    
        A simple helper function for making predictions, based on a loaded dataset.
    
    """


    # Create a predict graph - the backbone of the predict workflow
    predict_graph = Graph()

    # Construct the model name of interest
    model_name = f"fs_regressor_{iteration}"

    # Find the model ID corresponding to the model name
    for model in client.resources.list_resources(project_id=project_id, resource_type="model"):
        if model.name == model_name:
            break  # Stop after finding the first match. We can use the model ID for future reference

    input_space_id = client.resources.get_resource_id_by_name(

                        resource_type="dataset"
                        , project_id=project_id
                        , name=input_space_name

                )

    # Create node to load the model (using model ID)
    load_model = LoadModel(

        label="Load Model"
        , project_id=project_id
        , file_id=model.id
        , client=client

    )

    input_space = LoadDataset(

        label="Input Space"
        , project_id=project_id
        , file_id=input_space_id
        , client=client

    )

    predict = Node(

        node_name="PredictModel"
        , label="Predict Model"
        , dataset= input_space.make_handle("file")
        , model=load_model.make_handle("file")
        , client=client

    )

    # Add handles to the prediction and uncertainty outputs
    output_predictions = predict.make_handle("prediction")
    output_uncertainty = predict.make_handle("uncertainty")

    # Define download nodes for predictions and uncertainty
    save_predictions = Save(

        label="Save Predictions"
        , data=output_predictions
        , file_name=f"mean_{iteration}"
        , project_id=project_id
        , client=client

    )

    save_uncertainty = Save(

        label="Save Uncertainty"
        , data=output_uncertainty
        , file_name=f"std_{iteration}"
        , project_id=project_id
        , client=client

    )

    predict_graph.add_node(load_model)
    predict_graph.add_node(input_space)
    predict_graph.add_node(predict)
    predict_graph.add_node(save_predictions)
    predict_graph.add_node(save_uncertainty)

    predict_workflow = Workflow(

            graph=predict_graph.nodes
            , inputs=predict_graph.external_input
            , external_input_id=predict_graph.external_input_id
            , client=client

        )

    predict_response = client.run_node(predict_workflow)

    result = predict_response.status.value == "completed"
    
    if not result:
        pprint(predict_response.model_dump())
    
    assert predict_response.status.value == "completed"


####
5.
####

def get_presigned_url(url):

    """

        Get the contents from the presigned url.

    """
    url = url.replace("https://", "http://")
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response

####
6.
####

def recommend_new_sample(client, project_id, iteration, acquisition_function="PosteriorStandardDeviation"):

    """
    
        Recommend the next point to sample in the input space.
        NB: PosteriorStandardDeviation is chosen as the default acquisition function

    """

    recommend_graph = Graph()

    # Construct the model name of interest
    model_name = f"fs_regressor_{iteration}"

    # Find the model ID corresponding to the model name
    for model in client.resources.list_resources(project_id=project_id, resource_type="model"):
        if model.name == model_name:
            break  # Stop after finding the first match. We can use the model ID for future reference

    # Create node to load the model (using model ID)
    load_model = LoadModel(

        label="Load Model"
        , project_id=project_id
        , file_id=model.id
        , client=client

    )

    recommend_graph.add_node(load_model)

    recommend_model = Node(

        node_name="Recommend"
        , label="Recommend New Sample Point"
        , model=load_model.make_handle("file")
        , acquisition_function=acquisition_function
        # Constrained to one recommended dpa
        , number_of_points=1
        , client=client

    )

    download_rec = Download(

        label="Download Recommendation"
        , file=recommend_model.make_handle("recommended_points")
        , client=client
    )

    recommend_graph.add_node(recommend_model)
    recommend_graph.add_node(download_rec)

    workflow = Workflow(

        graph=recommend_graph.nodes
        , inputs=recommend_graph.external_input
        , external_input_id=recommend_graph.external_input_id
        , requested_output={

            "Recommendation": download_rec.make_handle("file").model_dump()

        }
        , client=client

    )

    recommend_response = client.run_node(workflow)
    
    result = recommend_response.status.value == "completed"
    
    if not result:
        pprint(recommend_response.model_dump())

    assert recommend_response.status.value == "completed"

    predictions_response = get_presigned_url(recommend_response.outputs["outputs"]["Recommendation"])    
    recommendation = float(pd.read_csv(StringIO(predictions_response.text)).iloc[0, 0])

    return recommendation
