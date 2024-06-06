# Load the prompt configuration
import yaml
import re
import torch
import random
import numpy as np
import os
from munch import Munch
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_and_tokenizer(configs,quantization_config):
    model_name = configs.model['name']
    access_token = configs.model['access_token']
    print(f"Downloading model {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    # model.cuda()
    # model = []
    return model, tokenizer


def load_config(path):
    # absolute_path = os.path.abspath(path)
    # print("Current Working Directory:", os.getcwd())
    # print("Absolute Path:", absolute_path)
    with open(path, "r") as f:
        mydict = yaml.safe_load(f)
    configs = Munch(mydict)
    return configs


def tokenize_function(example,tokenizer):
    # start_prompt = 'Summarize the following conversation.\n\n'
    # end_prompt = '\n\nSummary: '
    # prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]

    prompt = f"<s>[INST] {example['input_text'].strip()} [/INST]"
    input_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    example['input_ids'] = input_tokens
    example['labels'] = tokenizer(example["target_code"], return_tensors="pt").input_ids
    example['token_size'] = len(input_tokens[0])

    return example


# to get the error list
def get_committed_errors(df, error):
    if '_' in error:
        error_ids = [int(id) for id in error.split('_')]
    else:
        error_ids = [error]

    # Remove duplicates and convert the error_labels to follow the names from the dictionary error_indices
    error_ids = list(set(error_ids))
    error_ids = [int(id) for id in error_ids]
    errors_committed = [df.loc[df['key'] == int(id), 'description'].values[0] for id in error_ids]

    return errors_committed


def get_question_details(df, question_id):
    # Filter the DataFrame for the specific assignment and problem
    filtered_df = df[df['ProblemID'] == question_id]

    if filtered_df.empty:
        return "No data found for the given AssignmentID and ProblemID"

    # Get the question requirement
    requirement = filtered_df['Requirement'].iloc[0]

    # Get the question requirement
    method_head = filtered_df['Method'].iloc[0]

    # Get the list of concepts being tested (non-null columns after 'Requirement')
    concepts = [col for col in filtered_df.columns[3:] if filtered_df[col].notna().any()]

    return requirement, concepts, method_head


# Extract the code using triple backticks
def extract_code(response, method_header):
    pattern = pattern = re.compile(r"```(.*?)```(.*)", re.DOTALL)

    # pattern = re.compile(rf"{re.escape(method_header[:-2])}.*?\n\}}\s*```(.*)", re.DOTALL)
    match = pattern.search(response)

    if match:
        code = match.group(1).strip()
        explanation = match.group(2).strip()
    else:
        code = response
        explanation = ""
        print("**************this is empty **************")

    # Print the extracted code and explanation
    # print("method header:")
    # print(method_header)
    # print("\nCode:")
    # print(code)
    # print("\nExplanation:")
    # print(explanation)

    return code, explanation


def get_compiler_message(df, codeID):
    # message = [df['CodeStateID'] == codeID, 'CompileMessageData']
    messages = df.loc[df['CodeStateID'] == codeID, 'CompileMessageData'].dropna().values.tolist()
    if not messages:
        message = "No errors"
    else:
        message = messages

    # print("error_messages:", message)
    return message


def extract_code(response):

    pattern = pattern = re.compile(r"```(.*?)```(.*)", re.DOTALL)

    match = pattern.search(response)

    if match:
        code = match.group(1).strip()
        explanation = match.group(2).strip()
    else:
        code = ""
        explanation = ""
        print("**************this is empty **************")


    return code, explanation