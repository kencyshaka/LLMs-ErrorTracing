import os

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

    output_csv_file_path = f"prompts_{attempts}-attempts_score-{isScore}_error-{isError}_raw-{isRaw}_code-{isCode}_concept-{isConcept}_instruction-{isInstruction}.csv"
    file = os.path.join(directory_path, output_csv_file_path)
    print("reading from the following file ===", file)

    # loading the dataset
    df = pd.read_csv(file)

    return df, output_csv_file_path


def make_dataloader(file):
    print("make dataloader", file)

