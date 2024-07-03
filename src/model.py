
from util import extract_code,run_java_code_and_get_report, extract_error_info,get_errorID


def generate_text(prompt, pipe, configs, df_error):
    result = pipe(
                    prompt,
                    max_new_tokens=int(configs.max_output_tokens),  # Ensure this is an integer
                    do_sample=True,
                    top_k=10,
                    return_full_text=False,
            )
    
    # TODO need to put into its own function to format and process output
    code, explanation = extract_code(result[0]['generated_text'])
    
    # code below is for testing purpose
    """ 
public int greenTicket(int a, int b, int c)
{
    int result;
    
    if (a != b & a != c)
    {
        result = 0;
    }
    else if (a = b = c)
    {
        result = 20;
    }
    else if ((a = b && a != c) || (b = c && b != a) || (c = a && c != b))
    {
        result = 10;
    }
    return result;
}
    """
    compiler_report = run_java_code_and_get_report(code) # Run the Java code and get the compiler report
    error_info, error_count, error_class = extract_error_info(compiler_report, df_error) # Extract the error information
    

    print("\nExtracted Error Information:")
    print(f"\nTotal number of errors: {error_count},ErrorIDS: {error_class}, Error: {error_info} ")


    return code, explanation,error_info, error_count, error_class
