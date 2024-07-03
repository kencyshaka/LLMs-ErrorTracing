import os
import numpy as np
import pandas as pd


def read_data(config):
    directory_path = os.path.join("../dataset/prompt/", config['overview']['prediction'])
    isAttempts = config['history']['attempts']['max']
    max_step = config['history']['attempts']['steps']
    isCode = config['overview']['content_options']['code']
    isError = config['overview']['content_options']['errors']
    isScore = config['overview']['content_options']['scores']
    isConcept = config['history']['questions']['concept']
    isInstruction = config['overview']['prediction_instruction']
    isRaw = config['overview']['content_options']['raw_errors']
    if isAttempts:
        attempts = max_step
    else:
        attempts = "all"

    output_csv_file_path = f"t_prompts_{attempts}-attempts_score-{isScore}_error-{isError}_raw-{isRaw}_code-{isCode}_concept-{isConcept}_instruction-{isInstruction}.csv"
    file = os.path.join(directory_path, output_csv_file_path)
    print("reading from the following file ===", file)

    # loading the dataset
    df = pd.read_csv(file)

    return df, output_csv_file_path


def make_dataloader(file):
    #TODO: create a dataloader for fine-tuning
    print("make dataloader", file)


def get_errorID_embeddings(error_indices, error, configs):
    # Convert error to string if it's an integer
    if isinstance(error, int):
        error = str(error)

    if '_' in error:
        # Split the error and handle -1 separately
        error_parts = error.split('_')
        error_ids = [int(id) for id in error_parts if id != '-1']
    else:
        if error == '-1':
            error_ids = [0]
        else:
            error_ids = [error]

    # Remove duplicates and convert the error_labels to follow the names from the dictionary error_indices
    error_ids = list(set(error_ids))
    error_ids = [error_indices.get(int(id), None) for id in error_ids]

    # print("error indices", error_ids)
    # Create a vector of size 10 with 1s at the specified indices
    vector_size = configs.top_error_size
    vector = np.zeros((1, vector_size))
    vector[0, error_ids] = 1

    return vector.tolist()[0]
