# LLMs-ErrorTracing

This repository contains code for in-context learning and fine-tuning Large Language Models (LLMs) to generate code 
based on students’ mastery of programming concepts. We utilize a real student dataset from the [CSEDM Data Challenge](https://sites.google.com/ncsu.edu/csedm-dc-2021/dataset)

## In-Context Learning
To run the in-context learning, execute incontext.sh located src folder. The process includes the following steps:

1.	Prompt Creation: The script creates prompts and passes them through the LLM to generate code.
2.	Error Extraction: The generated code is compiled to extract the top ten common errors from the error report.
3.	Evaluation: The predictions for the first and overall attempts are evaluated, and results are logged using Weights & Biases.

## Fine-Tuning
To be added…
