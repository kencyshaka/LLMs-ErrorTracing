import torch
import time
import wandb
import pandas as pd
from transformers import BitsAndBytesConfig, pipeline
from util import *
from datetime import datetime
from data_loader import *
from model import *
from evaluation import performance_granular


wandb.login()

def main(configs, output_path, model_path, result_path, error_df):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_random_seed(configs.seed)

    quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
    )
    
    run = wandb.init(
        # Set the project where this run will be logged
        project="LLM Error Tracing",
        # Track hyperparameters and run metadata
        config={
            "AssID": configs.overview['assignment'],
            "Semester": configs.overview['season'],
            "Prediction": configs.overview['prediction'],
            "IsCode": configs.overview['content_options']['code'],
            "IsError": configs.overview['content_options']['errors'],
            "IsRawError": configs.overview['content_options']['raw_errors'],
            "IsScore": configs.overview['content_options']['scores'],
            "train": configs.model['train'],
            "fine_tune": configs.model['fine-tuned_model'],
            "max_output_tokens": configs.max_output_tokens,
        })

    ## load the dataset
    test_set, test_file = read_data(configs)

    if not configs.model['train']:
        if configs.model['fine-tuned_model']:
            print("inference with fine-tuned model")
        else:
            print("inference without fine-tuned model")
            
            model, tokenizer = load_model_and_tokenizer(configs, quantization_config)

            # load the input_text for test set
            input_text = test_set[:5]
            print("input_text", input_text.shape)
            tokenized_datasets = input_text.apply(lambda row: tokenize_function(row, tokenizer), axis =1)
            print("tokenized_datasets", tokenized_datasets.shape)

            pipe = pipeline(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                temperature=0.2,
                do_sample=True,
                # repetition_penalty=1.1,
                return_full_text=True,
                max_new_tokens= int(configs.max_output_tokens),
            )

            
            tokenized_datasets[["generated_code", "explanation","error_list", "error_count", "error_class"]] = tokenized_datasets["prompt"].apply(lambda x: pd.Series(generate_text(x, pipe, configs, error_df)))
            output_df = tokenized_datasets.drop(columns=['prompt','input_ids', 'labels', ])
            performance_granular(output_df, error_df, result_path, configs) # performing first attempt and overall evaluatioin


            print("The new test shape====", tokenized_datasets.shape)
            print("The new test shape:", tokenized_datasets.head())

            # save the output
            filename = os.path.join(output_path, test_file)
            print("file save at ---------------------------------", filename)
            # Save the updated DataFrame to a CSV file
            output_df.to_csv(filename, index=False)
            

    else:
        print("fine-tuning the model")
        print("model will be saved here: ", model_path)


if __name__ == '__main__':
    config_path = 'config.yaml'
    error_path = '../dataset/error_indices.csv'

    # Load configuration, code, error and question details
    config = load_config(config_path)
    error_df = pd.read_csv(error_path) 

    # define the folders
    output_path = os.path.join("../dataset/output/", config['overview']['prediction'])
    model_path = os.path.join("../model/", config['overview']['prediction'])
    result_path = os.path.join("../result/", config['overview']['prediction'])


    if output_path:
        os.makedirs(output_path, exist_ok=True)

    if model_path:
        os.makedirs(model_path, exist_ok=True)
    
    if result_path:
        os.makedirs(result_path, exist_ok=True)

    main(config, output_path, model_path, result_path, error_df)
