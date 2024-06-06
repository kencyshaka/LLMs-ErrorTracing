import torch
import time
from transformers import BitsAndBytesConfig, pipeline
from util import load_config, set_random_seed, load_model_and_tokenizer,tokenize_function, extract_code
from datetime import datetime
from data_loader import *


def main(configs, output_path, model_path):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_random_seed(configs.seed)

    quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
    )

    ## load the dataset
    test_set, test_file = read_data(configs)

    if not configs.model['train']:
        if configs.model['fine-tuned_model']:
            print("inference with fine-tuned model")
        else:
            print("inference without fine-tuned model")
            model, tokenizer = load_model_and_tokenizer(configs, quantization_config)

            # load the input_text for test set
            input_text = test_set[0:2]
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
                max_new_tokens=configs.max_output_tokens,
            )

            sequences = pipe(
                tokenized_datasets,
                max_new_tokens=configs.max_output_tokens,
                do_sample=True,
                top_k=10,
                return_full_text=False,
            )
            generated_code = []
            explanations = []
            for seq in sequences:
                code, explanation = extract_code(seq['generated_text'])

                # Append to rows
                generated_code.append(code)
                explanations.append(explanation)

            # updated the test_set
            df = test_set.iloc[:2].copy()

            df.loc[:, 'input_token_size'] = tokenized_datasets['token_size']
            df.loc[:, 'generated_code'] = generated_code
            df.loc[:, 'explanation'] = explanations

            print("The new test shape====", df.shape)
            print("The new test shape:", df.head())

            # save the output
            filename = os.path.join(output_path, test_file)
            print("file save at ---------------------------------", filename)
            # Save the updated DataFrame to a CSV file
            df.to_csv(filename, index=False)

    else:
        print("fine-tuning the model")
        print("model will be saved here: ", model_path)


if __name__ == '__main__':
    config_path = 'config.yaml'
    # Load configuration, code, error and question details
    config = load_config(config_path)
    # define the folders
    output_path = os.path.join("../dataset/output/", config['overview']['prediction'])
    model_path = os.path.join("../model/", config['overview']['prediction'])

    if output_path:
        os.makedirs(output_path, exist_ok=True)

    if model_path:
        os.makedirs(model_path, exist_ok=True)

    main(config, output_path, model_path)