from argparse import ArgumentParser
import json

import pandas as pd
from openai import OpenAI

parser = ArgumentParser()

parser.add_argument(
    '-p',
    '--path',
    type=str,
    required=True,
    help="Path to the JSON file that contains the JSON file containing \
        parameters for different stories."
)

parser.add_argument(
    '-s',
    '--save',
    type=str,
    required=False,
    default='two/prompt_result.csv',
    help='CSV filepath to save the result.'
)

args = parser.parse_args()

if __name__ == '__main__':
    client = OpenAI(api_key="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")

    with open(args.path, 'rb') as f:
        story_infos = json.load(f)

    instruction_template = """
Using the information provided to you, generate a short story for a typical
3-year-old using the guides provided to you. The "Summary" is a short summary of
the story to generate. The "Features" are the list of features the story should
have. The "Sentence" is a sentence that should appear somewhere in the story.
The "Words", are the list of wards that should be included in the story.


"""

    for i, info in enumerate(story_infos, 1):
        story_info_str = [f"'{k}': {v}" for k, v in info.items()]
        story_info_str = '\n'.join(story_info_str)

        instruction = instruction_template + story_info_str

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": instruction}
            ]
        )

        story_infos[i-1]['Completion'] = response.choices[0].message.content

    df = pd.DataFrame(story_infos)
    df.to_csv(args.save, index=False)