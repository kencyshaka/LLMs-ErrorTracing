import pandas as pd
import itertools
import os
from util import load_config, get_question_details, get_committed_errors, get_compiler_message


# Function to generate the cumulative input text
def generate_input(submissions, question_numbers, scores, errors, dataframes, config):
    language = config['overview']['language']
    season = config['overview']['season']

    code_df = dataframes[0]
    question_df = dataframes[1]
    error_df = dataframes[2]
    main_df = dataframes[3]

    content_options = []
    if config['overview']['content_options']['code']:
        content_options.append("code submitted")
    if config['overview']['content_options']['errors']:
        content_options.append("errors encountered")
    if config['overview']['content_options']['scores']:
        content_options.append("scores")

    num_questions = len(set(question_numbers))
    question_references = f'question '
    question_references += ', '.join([f" {i}" for i in range(1, num_questions + 1)])

    text = f"I have a detailed history of a student's performance in a {language} introductory programming course during {season} 2019.\nIt includes the {', '.join(content_options)} received for multiple attempts on {question_references}.\n"
    text += f"Below are the detailed records of their submissions:\n"

    current_question = question_numbers[0]
    text += f"### Question "
    if config['history']['questions']['id']:
        text += f"{int(current_question) + 1} \n"
    if config['history']['questions']['text']:
        text += f"{get_question_details(question_df, current_question)[0]}\n"
    if config['history']['questions']['concept']:
        text += f"Concept: {get_question_details(question_df, current_question)[1]}\n"
    if config['history']['questions']['method_head']:
        text += f"Given method signature: {get_question_details(question_df, current_question)[2]}\n"
    attempt_number = 1

    for i in range(len(submissions) - 1):
        next_question = question_numbers[i + 1]
        if question_numbers[i] != current_question:
            current_question = question_numbers[i]
            text += f"### Question "
            if config['history']['questions']['id']:
                text += f"{int(current_question) + 1} \n"
            if config['history']['questions']['text']:
                text += f"{get_question_details(question_df, current_question)[0]}\n"
            if config['history']['questions']['concept']:
                text += f"Concept: {get_question_details(question_df, current_question)[1]}\n"
            if config['history']['questions']['method_head']:
                text += f"Given method signature: {get_question_details(question_df, current_question)[2]}\n"
            attempt_number = 1

        text += f"Attempt {attempt_number}:\n"
        if config['overview']['content_options']['code']:
            text += f"Code: {code_df.loc[code_df['CodeStateID'] == submissions[i], 'Code'].values[0]}\n"
        if config['overview']['content_options']['errors'] and not config['overview']['content_options']['raw_errors']:
            text += f"Errors: {get_committed_errors(error_df, errors[i])}\n"
        elif config['overview']['content_options']['errors'] and config['overview']['content_options']['raw_errors']:
            text += f"Errors: {get_compiler_message(main_df, submissions[i])}\n"
        if config['overview']['content_options']['scores']:
            text += f"Score: {'Pass' if scores[i] == 1 else 'Fail'}\n\n"
        attempt_number += 1

    if config['overview']['prediction_instruction']:
        text += "### Instructions for Prediction:\n"
        if config['instructions']['options']['one']:
            text += f"Progression and Learning: Consider the student's progression in terms of scores, complexity of the code over time.\n"
        if config['instructions']['options']['two'] and config['overview']['content_options']['code']:
            text += f"Patterns in Code: Examine common structures, logic, and syntax used by the student.\n"
        if config['instructions']['options']['three'] and config['overview']['content_options']['errors']:
            text += f"Error Correction: Look at how the student corrected their errors over multiple attempts and improved their code.\n"
        if config['instructions']['options']['four']:
            text += f"Style and Approach: Maintain the student's style and approach as seen in their previous submissions.\n\n"

    if current_question == next_question:
        next_attempt = attempt_number
        next_task = f"Question {current_question + 1}, Attempt {next_attempt}"

    else:
        next_task = f"The next question {ques[j + 1] + 1}"
        if config['output']['next_task']['text']:
            next_task += f":\n'{get_question_details(question_df, current_question + 1)[0]}'"
        if config['output']['next_task']['concept']:
            next_task += f"\nEvaluated concepts: {get_question_details(question_df, current_question + 1)[1]}"

    if config['overview']['prediction'] == 'error':
        next_task += f"\nHere is the list of possible errors: {error_df['description'].tolist()}"
    else:
        next_task += f"\nMake sure to use this method signature {get_question_details(question_df, current_question + 1)[2]}"

    considerations = ""
    if config['output']['next_task']['considerations']['one']:
        considerations += "Align the prediction with the student's coding style and understanding.\n"
    if config['output']['next_task']['considerations']['two']:
        considerations += "Provide a brief explanation of the key factors considered."

    text += f"\n### Output:\nBased on this history, predict the {'code' if config['overview']['prediction'] == 'code' else 'errors if any'} that the student will {'write' if config['overview']['prediction'] == 'code' else 'encounter'} for \n{next_task} \n{considerations}\n"
    if config['overview']['prediction'] == 'code':
        text += "### Code: #write the predicted code here\n### Explanation: # write the explanation here"
    else:
        text += "### Errors:\n### Explanation: "

    return text


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_path = '../dataset/test_set.csv'
    config_path = 'config.yaml'
    error_path = '../dataset/error_indices.csv'
    code_path = '../dataset/CodeStates.csv'
    question_path = '../dataset/prompt_concept.csv'
    main_path = '../dataset/MainTable.csv'
    subject_path = '../dataset/Subject.csv'
    save_prompts = '../dataset/prompt/'

    # Load configuration, code, error and question details
    config = load_config(config_path)

    question_df = pd.read_csv(question_path)
    question_df = question_df[question_df['AssignmentID'] == config['overview']['assignment']]
    question_df['ProblemID'] = range(len(question_df))
    print(question_df.shape)
    get_question_details(question_df, 0)

    error_df = pd.read_csv(error_path)
    code_df = pd.read_csv(code_path)
    grade_df = pd.read_csv(subject_path)
    main_df = pd.read_csv(main_path)
    print("mainTable", main_df.shape)

    print("error", error_df['description'].tolist())

    directory_path = os.path.join(save_prompts, config['overview']['prediction'])
    # creating saving directory
    if save_prompts:
        os.makedirs(directory_path, exist_ok=True)

    dataframes = [code_df, question_df, error_df, main_df]
    # Read the file and process each student's data
    count = 0
    rows = []
    isAttempts = config['history']['attempts']['max']
    max_step = config['history']['attempts']['steps']

    with open(test_path, 'r') as file:
        for lent, css, ques, ans, err in itertools.zip_longest(*[file] * 5):
            lent = int(lent.strip().strip(','))
            css = [cs for cs in css.strip().strip(',').split(',')]
            ques = [int(q) for q in ques.strip().strip(',').split(',')]
            ans = [int(a) for a in ans.strip().strip(',').split(',')]
            err = [er for er in err.strip().strip(',').split(',')]
            print(lent)

            # if count == 12:
            if isAttempts and lent >= max_step:
                steps = max_step
                ques = ques[-steps:]
                ans = ans[-steps:]
                css = css[-steps:]
                err = err[-steps:]

            else:
                steps = lent
            
            #the the student ID and their final grade
            user_id = main_df.loc[main_df['CodeStateID'] == css[0], 'SubjectID'].values[0]
            if not grade_df.loc[grade_df['SubjectID'] == user_id, 'X-Grade'].empty:
                grade = grade_df.loc[grade_df['SubjectID'] == user_id, 'X-Grade'].values[0]
            else:
                grade = "not found"

            print("user_id",user_id)
            print("grade",grade)

            if count < 10: # to select a portion of the testset
                for j in range(0, steps - 1):
                    # Include all previous submissions up to the current one
                    current_submissions = css[:j + 2]
                    current_question_numbers = ques[:j + 2]
                    current_scores = ans[:j + 2]
                    current_errors = err[:j + 2]
                    current_error_message = get_compiler_message(main_df, current_submissions[-1])

                    input_text = generate_input(current_submissions, current_question_numbers,
                                            current_scores, current_errors, dataframes, config)
                    
                    rows.append({"user_id": user_id,
                                 "grade": grade,
                             "problem_id": current_question_numbers[-1],
                             "input_text": input_text,
                             "target_errors": current_errors[-1],
                             "raw_target_errors": current_error_message,
                             "target_code": code_df.loc[code_df['CodeStateID'] == current_submissions[-1], 'Code'].values[0]})

                    
            count += 1

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(rows)
    print("shape of the df", df.shape)
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
    save_file = os.path.join(directory_path, output_csv_file_path)
    df.to_csv(save_file, index=False)

    print(f"CSV file saved as: {save_file}")
