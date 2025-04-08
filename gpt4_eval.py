from argparse import ArgumentParser
import json
import logging
import os
from random import randint
import warnings
import requests

from datasets import load_dataset
#from llama_cpp import Llama
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer,
                          GPT2Tokenizer,
                          GPT2LMHeadModel)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.disable(logging.WARNING)

parser = ArgumentParser()

parser.add_argument(
    '-m',
    '--model',
    type=str,
    required=True,
    help="The model that will process the individual texts in the dataset. All \
        possible models are: TS-1M,TS-3M,TS-8M,TS-28M,TS-33M,TS-1Layer-21M,\
        GPT2-XL,LLaMa-7B"
        # TS-1Layer,TS-2Layer,TS-Instruct."
)

parser.add_argument(
    '-n',
    '--num',
    type=int,
    required=True,
    help="The number of texts to test for in the dataset."
)

parser.add_argument(
    '-d',
    '--device',
    type=str,
    choices=['cpu', 'cuda', 'mps'],
    default='cpu',
    required=False,
    help="The device to run the model on, could be cpu, cuda (NVIDIA GPU) or \
        mps (Apple Silicon)."
)

parser.add_argument(
    '-s',
    '--save',
    type=str,
    required=False,
    default='prompt_result.csv',
    help="CSV filename to save the result."
)

args = parser.parse_args()

ts_models = {
    'TS-1M': 'roneneldan/TinyStories-1M',
    'TS-3M': 'roneneldan/TinyStories-3M',
    'TS-8M': 'roneneldan/TinyStories-8M',
    'TS-28M': 'roneneldan/TinyStories-28M',
    'TS-33M': 'roneneldan/TinyStories-33M',
    'TS-1Layer-21M': 'roneneldan/TinyStories-1Layer-21M'
    # 'TS-1Layer',
    # 'TS-2Layer',
    # 'TS-Instruct'
}

gpt_models = {
    'GPT2-XL': 'gpt2-xl',
    'GPT2-small': 'gpt2',
    'GPT2-medium': 'gpt2-medium'

}

# llama_models = {
#     'LLaMa-7B': '../../../../Projects/LLMs/llama/llama-2-7b/ggml-model-f16_q4_0.bin',
#     'LLaMa-13B': '../../../../Projects/LLMs/llama/llama-2-13b-chat/ggml-model-q4_0.bin'
# }

llama_models = {
    "LLaMa-7B": "llama2",
    "TinyLLaMa": "tinyllama"
}

all_models = list(ts_models) + list(gpt_models) + list(llama_models)

gpt4_instructions = {
    "first_instruction": "the following exercise, the student is given a \
beginning of a story. The student needs to complete it into a full story. The \
exercise tests the student ́s language abilities and creativity. The symbol *** \
marks the separator between the prescribed beginning and the student’s \
completion:",

    "second_instruction": "Please provide your general assessment about the \
part written by the student (the one after the *** symbol). Is it gramatically \
correct? Is it consistent with the beginning of the story? Pay special \
attention to whether the student manages to complete the sentence which is \
split in the middle by the separator ***.",

    "third_instruction": "Now, grade the student’s completion in terms of \
grammar, creativity, consistency with the story’s beginning and whether the \
plot makes sense. Moreover, please provide your best guess of what the age of \
the student might be, as reflected from the completion. Choose from possible \
age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16."
}

if __name__ == '__main__':

    client = OpenAI(api_key="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")

    if args.model not in all_models:
        raise ValueError(f"{args.model} is not a valid model. The only valid models \
                            are {','.join(all_models)}")
    
    if args.num <= 0 or type(args.num) != int:
        raise ValueError(f"{args.num} should not be zero or negative, and must\
                         be an integer.")
    
    dataset = load_dataset("roneneldan/TinyStories")
    dataset = dataset['validation']

    prompts = []

    for i, data in enumerate(dataset):
        if i == args.num:
            break

        story = data['text']
        story_space_split = story.split(' ')

        forty_perc = int(len(story_space_split) * 0.4)
        sixty_perc = int(len(story_space_split) * 0.6)

        cutoff_index = randint(forty_perc, sixty_perc)
        prompt = ' '.join(story_space_split[:cutoff_index])
        prompts.append(prompt)
        
    m = args.model

    results = {'prompts': prompts,
               'completions': [],
               'full_texts': [],
               'feedbacks': [],
               'grammar': [],
               'creativity': [],
               'consistency': [],
               'age_group': []}

    if m in ts_models:
        model_name = ts_models[m]
        model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token

        for prompt in tqdm(prompts, desc=f'Running {m} model for completions...'):
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
            output = model.generate(input_ids, 
                                    max_length = 1000, 
                                    num_beams=1,
                                    do_sample=False) # temperature=0
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)

            prompt_length = len(prompt)
            completion = output_text[prompt_length:].strip()

            results['completions'].append(completion)
            results['full_texts'].append(prompt + '*** ' + completion)

    elif m in gpt_models:
        model_name = gpt_models[m]
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name).to(args.device)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token

        for prompt in tqdm(prompts, desc=f'Running {m} model for completions...'):
            input_ids = tokenizer(prompt, 
                                    padding=True, 
                                    return_tensors='pt'
                                    ).to(args.device)
            
            output = model.generate(**input_ids, 
                                    max_length=1000,
                                    do_sample=False) # temperature=0

            output_text = tokenizer.batch_decode(output, 
                                                    skip_special_tokens=True)
            output_text = output_text[0]

            prompt_length = len(prompt)
            completion = output_text[prompt_length:].strip()

            results['completions'].append(completion)
            results['full_texts'].append(prompt + '*** ' + completion)

    elif m in llama_models:
        # model_path = llama_models[m]

        # model = Llama(model_path = model_path,
        #                 n_ctx = 2048,            # context window size
        #                 n_gpu_layers = 1,        # enable GPU
        #                 use_mlock = True,        # enable memory lock so not swap
        #                 verbose=False)
        
        # for prompt in tqdm(prompts, desc=f'Running {m} model for completions...'):
        #     output = model(prompt=prompt, max_tokens=120, temperature=0.01)
        #     completion = output['choices'][0]['text'].strip()

        model_name = llama_models[m]
        for prompt in tqdm(prompts, desc=f'Running {m} model...'):

            response = requests.post(
                url="http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt
                }
            )

            res_dec = response.content.decode()

            completion = ''
            for line in res_dec.split('\n')[:-1]:
                line_json = json.loads(line)
                t = line_json["response"]
                completion = completion + t

            results['completions'].append(completion)
            results['full_texts'].append(prompt + '*** ' + completion)

    for i, prompt in tqdm(enumerate(results['prompts']), desc="Running GPT4 evaluations..."):

        response = client.chat.completions.create(
            model='gpt-4',
            temperature=0,
            messages=[
                {"role": "system", "content": gpt4_instructions['first_instruction']},
                {"role": "user", "content": results['full_texts'][i]},
                {"role": "system", "content": gpt4_instructions['second_instruction']},
            ]
        )

        feedback = response.choices[0].message.content
        results['feedbacks'].append(feedback)

        response = client.chat.completions.create(
            model='gpt-4',
            temperature=0,
            messages=[
                {"role": "system", "content": gpt4_instructions['first_instruction']},
                {"role": "user", "content": results['full_texts'][i]},
                {"role": "system", "content": gpt4_instructions['second_instruction']},
                {"role": "assistant", "content": feedback}
            ],
            function_call={'name': 'grade_grammar_creativity_consistency_agegroup'},
            functions = [
                {
                    "name": "grade_grammar_creativity_consistency_agegroup",
                    "description": gpt4_instructions['third_instruction'],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "grammar": {
                                "type": "integer",
                                "description": "A rating of the student's grammer as a value from 1 to 10."
                            },
                            "creativity": {
                                "type": "integer",
                                "description": "A rating of the student's creativity as a value from 1 to 10."
                            },
                            "consistency": {
                                "type": "integer",
                                "description": "A rating of the student's consistency as a value from 1 to 10."
                            },
                            "age_group": {
                                "type": "string",
                                "description": "A rating of the student's age group as a letter from A to F."
                            }
                        },
                        "required": ["grammar", "creativity", "consistency", "age_group"]
                    }
                }
            ]
        )

        grading = response.choices[0].message.function_call.arguments #['choices'][0]['message']['function_call']['arguments']

        grading = json.loads(grading)

        results['grammar'].append(grading['grammar'])
        results['creativity'].append(grading['creativity'])
        results['consistency'].append(grading['consistency'])
        results['age_group'].append(grading['age_group'])

    result = pd.DataFrame(results)
    print(result)

    if args.save.endswith('.csv'):
        csv_filename = args.save
    else:
        csv_filename = 'prompt_result.csv'
        
    result.to_csv(csv_filename, index=False)
    print(f'\n\nResult saved to {csv_filename}')