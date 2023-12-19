import pandas as pd
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--start", type=int, help="start index")
parser.add_argument("--end", type=int, help="end index")
parser.add_argument("--device", type=str, help="device_id (cuda:0)")

args = parser.parse_args()

start = args.start
end = args.end
device = args.device


#load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Orca-2-7b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_size = "right"
model.to(device)


#load the file
df = pd.read_csv('gsm_eval_set.csv')
questions = df['question'].to_list()
questions = questions[start:end]

# def extract_final_answer(s):
#     # Create a pattern that matches 'Final answer :' followed by any characters and then digits
#     pattern = r'Final answer\s*:\s*.*?(\d+)'
    
#     # Search the string for the pattern
#     match = re.search(pattern, s)
    
#     # If a match was found, return the digits
#     if match:
#         return match.group(1)
    
#     # If no match was found, return None
#     return None

def inf(model,tokenizer,question,device):
    
    system_message = """Follow these instructions : \n
        1)At the end of solution give final answer value exactly like this #### Final Answer : <answer value>\n
    
        """

    test_prompt = f"""<|im_start|>system\n{system_message}<|im_end|>\n
                     <|im_start|>user\n{question}<|im_end|>\n
                     <|im_start|>assistant"""
    model_inputs = tokenizer(test_prompt, return_tensors='pt').to(device)
    greedy_output = model.generate(**model_inputs, max_new_tokens=1500)
    out = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    #print(extract_final_answer(out))
    return question,out

answers = []
query = []

for q in questions:
    ques,answer = inf(model,tokenizer,q,device)
    print(ques)
    print("-"*100)
    print(answer)
    print("*"*100)
    answers.append(answer)
    query.append(ques)


result_df = pd.DataFrame({
            'question': query,
            'generated_answer': answers
        })

result_df.to_csv(f'Orca2_7b_GSM_Evaluation_{start}_{end}.csv', index=False)

# answers = []
# query = []

# for q in questions:
#     ans_list = [] # a list to store each response from the model for a particular question
#     print("*"*50)
#     print(q)
#     ques = None
#     for i in range(8): # loop to run your inf function on the same question 5 times
#         ques, answer = inf(model, tokenizer, q, device)
#         print("*"*50)
#         print(answer)
#         ans_list.append(answer) # appending each answer to the ans_list
#     answers.append(ans_list) # appending the list of answers to the main answers list
#     query.append(ques)

# # Now, your 'answers' list is a list of lists, where each sublist has 5 answers related to the same question.

# # you can now turn these into a pandas DataFrame like so:
# import pandas as pd

# result_df = pd.DataFrame(query, columns=['question'])

# # we separate answers into individual columns
# result_df[['answer1', 'answer2', 'answer3', 'answer4', 'answer5','answer6', 'answer7', 'answer8']] = pd.DataFrame(answers)

# result_df.to_csv(f'GSM_Evaluation_{start}_{end}.csv', index=False)



    