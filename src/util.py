# Load the prompt configuration
import yaml
import re
import torch
import random
import numpy as np
import os
import subprocess
from munch import Munch
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_random_seed(seed):
    # Set seed for built-in random module
    random.seed(seed)
    
    # Set seed for numpy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for Hugging Face Transformers
    from transformers import set_seed
    set_seed(seed)
    
    # Set PYTHONHASHSEED environment variable to ensure hash-based functions are deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_model_and_tokenizer(configs,quantization_config):
    model_name = configs.model['name']
    access_token = configs.model['access_token']
    print(f"Downloading model {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    
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
    # TODO: prepare the prompts based on the LLM, currently using codeLlama format

    prompt = f"<s>[INST] {example['input_text'].strip()} [/INST]"
    input_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    example['prompt'] = prompt
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



def get_errorID(df_error, error):
    
    error = error.lstrip()
    # Clean the error message
    clean_error = replace_msg_identifiers(error)
    print("The error after cleaning----", clean_error)
    
    # Try to find the row with the cleaned error message
    try:
        row = df_error[df_error['message'] == clean_error].iloc[0]
        errorID = row['value']
    except IndexError:
        # If the row does not exist, assign -1 to errorID
        errorID = -1
        print("************** No matching row found **************")

    return errorID



def extract_code(response):

    pattern = pattern = re.compile(r"```(.*?)```(.*)", re.DOTALL)

    match = pattern.search(response)

    if match:
        code = match.group(1).strip()
        # Remove everything before the word "public"
        public_pattern = re.compile(r"(public.*)", re.DOTALL)
        public_match = public_pattern.search(code)
        # print("code before======",code) 
        if public_match:
            code = public_match.group(1).strip()
            # print("remove the code-----",code)
            # print("removing anything before the public***********************************")        
        explanation = match.group(2).strip()
    else:
        code = ""
        explanation = response
        print("**************this is empty **************")


    return code, explanation



def extract_error_info(error_report,df_error):

    print("error_report*******************", error_report.strip())
    if not error_report.strip():
        return ["No errors"], 0, 0

        # Regular expression to match errors
    
    error_pattern = re.compile(r"(.*?):(\d+): error: ([^\n]*)")
    matches = error_pattern.findall(error_report)

    errors = []
    error_ids = []
    for match in matches:
        _, line_number,error_message = match

        s = error_message.split(':')
        if (len(s) == 2) and not any(word in s[1] for word in ["variable", "method", "class"]):
            error_message = s[0]
            print("s[0]",s[0])
            print("s[1]",s[1])

        # Check if the error message contains "cannot find symbol"
        if "cannot find symbol" in error_message:
           # Extract the symbol details from the report
            symbol_pattern = re.compile(r"symbol:\s+(.*?)\s*\n\s*location:", re.DOTALL)
            symbol_match = symbol_pattern.search(error_report)

            if symbol_match:
                symbol_details = symbol_match.group(1).strip()
                error_message = f"{error_message.split('symbol')[0].strip()} symbol: {symbol_details}"
        
        error_ids.append(get_errorID(df_error,error_message))    
        errors.append(f"Line {line_number}:{error_message}")
    
    error_class = '_'.join(str(id) for id in error_ids)
    error_count = len(matches)
    
    return errors, error_count, error_class 


def run_java_code_and_get_report(code, file_name="TempJavaCode"):
    # Wrap the code in a complete Java class
    class_name = file_name
    wrapped_code = f"""
    public class {file_name} {{
        {code}

        public static void main(String[] args) {{
            // {class_name} instance = new {class_name}();
            // Add a call to the generated method here, adjust parameters as necessary
            // System.out.println(instance.in1To10(5, true)); // Example call
        }}

        
    }}
    """

    try:
        # Save the Java code to a temporary file
        java_file = f"{file_name}.java"
        with open(java_file, "w") as f:
            f.write(wrapped_code)

        # Compile the Java code
        compile_result = subprocess.run(
            ["javac", "--release", "8", java_file], # ["javac", java_file],
            capture_output=True,
            text=True
        )

        # Check if there were any compilation errors
        if compile_result.returncode != 0:
            # There were compilation errors, return the stderr output
            error_messages = compile_result.stderr.strip()

            return compile_result.stderr.strip()

        # Run the compiled Java code
        run_result = subprocess.run(
            ["java", file_name],
            capture_output=True,
            text=True
        )
        
        print("the error report",run_result)
        # Check if there were any runtime errors
        if run_result.returncode != 0:
           return run_result.stderr.strip()
        else:
            # No errors, return stdout output
            
            return run_result.stdout.strip()

    except Exception as e:
        
        return str(e)

    finally:
        # Clean up the temporary files
        try:
            os.remove(java_file)
            os.remove(f"{file_name}.class")
        except Exception as e:
            pass


def get_error_message(error):
    # some are just 3 and some just 2
    s = error.split(':')
    if (len(s) == 4) and any(word in s[3] for word in ["variable", "method", "class"]):
        last_two = ":".join(s[-2:])  # joins the last two parts using ":"
        msg = last_two
    elif ((len(s) == 4) and (len(s[2]) == 1)):
        last_two = "something ".join(s[-2:])  # joins the last two parts using ":"
        msg = last_two
    elif ((len(s) >= 3) and (len(s[2]) != 1)):
        msg = s[2]
    elif (len(s) == 2 and s[1] != "error"):
        msg = s[1]

    return msg


def replace_msg_identifiers(msg):
    # Define a regular expression pattern to match all special characters
    pattern1 = r"(?<=')[^a-zA-Z0-9\s]+(?=')"
    # Define a regular expression pattern to match the phrase for variable, method, class, package names
    pattern2 = r"variable\s+(.*?)\s+might not have been initialized"
    pattern3 = r"variable\s+(.*?)\s+already defined in method\s+(.*?)$"
    pattern4 = r"no suitable method found for\s+(.*?)$"
    pattern5 = r"method\s+(.*?)\s+already defined in class\s+(.*?)$"
    pattern6 = r"method\s+(.*?)\s+in class\s+(.*?)\s+cannot be applied to given types;"
    pattern7 = r"non-static method\s+(.*?)\s+cannot be referenced from a static context"
    pattern8 = r"package\s+(.*?)\s+does not exist"
    pattern9 = r"variable\s+(.*?)\s+already defined in class\s+(.*?)$"
    pattern10 = r"bad operand type\s+(.*?)\s+for unary operator 'ID'"
    pattern11 = r"no suitable constructor found for\s+(.*?)$"
    pattern12 = r"class\s+(.*?)\s+already defined in package unnamed\s+(.*?)$"
    pattern13 = "cannot assign a value to final variable\s+(.*?)$"
    pattern14 = "> expected"
    pattern15 = "< expected"
    pattern16 = "-> expected"
    pattern17 = "'ID' or 'ID' expected"
    pattern18 = "-'ID' expected"
    pattern19 = "Illegal static declaration in inner class\s+(.*?)$"
    pattern20 = ": expected"
    pattern21 = r"cannot find symbol: variable\s+(\S+)\s*" # r"cannot find symbol: variable\s+(.*?)$"
    pattern22 = "cannot find symbol: method\s+(.*?)$"
    pattern23 = "cannot find symbol: class\s+(.*?)$"

    # Define the replacement text
    replace_text2 = "variable ID might not have been initialized"
    replace_text3 = "variable ID already defined"
    replace_text4 = "no suitable method found for method ID"
    replace_text5 = "method ID already defined in class ID"
    replace_text6 = "method ID in class ID cannot be applied to given types"
    replace_text7 = "non-static method ID cannot be referenced from a static context"
    replace_text8 = "package ID does not exist"
    replace_text9 = "bad operand type for unary operator 'ID'"
    replace_text10 = "no suitable constructor found"
    replace_text11 = "class ID already defined in package unnamed"
    replace_text12 = "cannot assign a value to final variable ID"
    replace_text13 = "'ID' expected"
    replace_text14 = "Illegal static declaration in inner class ID"
    replace_text15 = "cannot find symbol: variable ID"
    replace_text16 = "cannot find symbol: method ID"
    replace_text17 = "cannot find symbol: class ID"

    print("error mid before",msg)
    # Replace all special characters in the 'text' column with a token ID of your choice (in this case, 999)
    msg = re.sub(pattern1, 'ID', msg)
    
    print("error mid after",msg)
    # Replace the value the variable names with ID
    msg = re.sub(pattern2, replace_text2, msg)
    msg = re.sub(pattern3, replace_text3, msg)
    msg = re.sub(pattern4, replace_text4, msg)
    msg = re.sub(pattern5, replace_text5, msg)
    msg = re.sub(pattern6, replace_text6, msg)
    msg = re.sub(pattern7, replace_text7, msg)
    msg = re.sub(pattern8, replace_text8, msg)
    msg = re.sub(pattern9, replace_text3, msg)
    msg = re.sub(pattern10, replace_text9, msg)
    msg = re.sub(pattern11, replace_text10, msg)
    msg = re.sub(pattern12, replace_text11, msg)
    msg = re.sub(pattern13, replace_text12, msg)
    msg = re.sub(pattern14, replace_text13, msg)
    msg = re.sub(pattern15, replace_text13, msg)
    msg = re.sub(pattern16, replace_text13, msg)
    msg = re.sub(pattern17, replace_text13, msg)
    msg = re.sub(pattern18, replace_text13, msg)
    msg = re.sub(pattern20, replace_text13, msg)
    msg = re.sub(pattern19, replace_text14, msg)
    msg = re.sub(pattern21, replace_text15, msg)
    msg = re.sub(pattern22, replace_text16, msg)
    msg = re.sub(pattern23, replace_text17, msg)

    return msg        
