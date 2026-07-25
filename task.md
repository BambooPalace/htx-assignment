create a repo with below
- make two mock endpoint /chat,  /model
    - /chat is a end to end llm system, it accepts {input: string}, returns answer as output with intermediate data{output:str, query:str, contexts:list[str]}
    it return dummy reponse, with output a random value of expected fields from below test data expected field. query same as input, contexts as [output]

    - /model is a LLM model only, its inputs and outputs is OPENAI API compatible.
    it return randomly 'true', 'false', 'mock output' as dummy response when called.

- generate a dumpy jsonl file as data/test.jsonl, sample as below, generate 50 samples
{"id": "q1", "input": "What is the leave policy?", "expected": "14 days annual leave"}
{"id": "q2", "input": "Who approves travel claims?", "expected": "Direct manager"}

- create a predict.py
    - for each test data, it call /chat with test input and log the output & contexts in the dict.
    - the new data is saved as data/outputs.jsonl

- create evaluate.py
    - for each data in outputs.jsonl, call /model with a prompt template to classify if the output is accurate answer given input and expected answer. /model supposed to return true or false label only. add exception handling for malformed output strings, and log as error.
    - set temperature as o, add a parameter n (n=3) to run model n times on each eval.
    - after evals are ran, record the accuracy, and use majority of votes from the n runs.
    - if data that /chat failed to answer correctly, ask /model to examine reason of failure given input, expected, and the /chat outputs and query and contexts; and generate a failure with reasons report. this function name as def diagnose
    - log exceptions and progress bar and n etc in terminal outputs as well as save as logs/log_ddmmss.text
    - when run evaluate.py, it should run the whole evaluate pipeline with args: n_runs, n_samples
- add unit test to make sure endpoints input output formats are correct and other functions work as expected.
- generate a README on what it does, how to run it


Notes:
- /chat return intermediate data such as contexts and query here, but these are typically not returned for a production api. Since we need find these data through logging in actually debugging for failure cases, for this demo and dummy api, i just return the intermediate as api output for pipeline convenience. To show how /model as a judge use these data, it s not a representation of real case.