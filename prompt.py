from argparse import ArgumentParser
import logging
import warnings
import requests
import json

# from llama_cpp import Llama
import pandas as pd
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer,
                          GPT2Tokenizer,
                          GPT2LMHeadModel)

logging.disable(logging.WARNING)

parser = ArgumentParser()

parser.add_argument(
    '-p',
    '--path', 
    type=str,
    required=True,
    help="Path to txt file containing the prompts to pass to the models. The \
        file can contain multiple prompts, but each prompt should be separated \
        from the others with three new lines (\\n\\n\\n). Meaning, there has to \
        be two consecutively empty lines between each prompt.",
)

parser.add_argument(
    '-m',
    '--models',
    type=str,
    required=True,
    help="The list of models that will process individual prompts. The list of \
        models should be listed without spaces, only with commas ','. All \
        possible models are: TS-1M,TS-3M,TS-8M,TS-28M,TS-33M,TS-1Layer-21M,\
        GPT2-XL,LLaMa-7B"
        # TS-1Layer,TS-2Layer,TS-Instruct."
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
    default='one/outputs/prompt_result.csv',
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

if __name__ == '__main__':
    with open(args.path, 'r') as f:
        prompts = f.read().split('\n\n\n')
        prompts = [p.strip() for p in prompts]
        prompts = [p for p in prompts if len(p) != 0]
    
    prompt_models = args.models.split(',')

    for m in prompt_models:
        if m not in all_models:
            raise ValueError(f"{m} is not a valid model. The only valid models \
                             are {','.join(all_models)}")
        
    result = {'Prompt': prompts}
        
    for m in prompt_models:
        result[m] = []

        if m in ts_models:
            model_name = ts_models[m]
            model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = tokenizer.eos_token

            for prompt in tqdm(prompts, desc=f'Running {m} model...'):
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
                output = model.generate(input_ids, 
                                        max_length = 1000, 
                                        num_beams=1,
                                        do_sample=False) # temperature=0
                output_text = tokenizer.decode(output[0], skip_special_tokens=True)

                prompt_length = len(prompt)
                completion = output_text[prompt_length:].strip()

                result[m].append(completion)

        elif m in gpt_models:
            model_name = gpt_models[m]
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name).to(args.device)
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = tokenizer.eos_token

            for prompt in tqdm(prompts, desc=f'Running {m} model...'):
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

                result[m].append(completion)

        elif m in llama_models:

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

                result[m].append(completion)



            # model_path = llama_models[m]

            # model = Llama(model_path = model_path,
            #                 n_ctx = 2048,            # context window size
            #                 n_gpu_layers = 1,        # enable GPU
            #                 use_mlock = True,        # enable memory lock so not swap
            #                 verbose=False)
            
            # for prompt in tqdm(prompts, desc=f'Running {m} model...'):
            #     output = model(prompt=prompt, max_tokens=120, temperature=0.01)
            #     completion = output['choices'][0]['text'].strip()

            #     result[m].append(completion)

    result = pd.DataFrame(result)
    print(result)

    result.to_pickle('prompt_result.pkl')
    print(f'\nResult saved as pandas DataFrame to prompt_result.pkl')

    if args.save.endswith('.csv'):
        csv_filename = args.save
    else:
        csv_filename = 'prompt_result.csv'
        warnings.warn(f'Invalid filename {args.save}. File will be saved as {csv_filename}', 
                      ValueError)
        
    result.to_csv(csv_filename, index=False)
    print(f'\n\nResult saved to {csv_filename}')
