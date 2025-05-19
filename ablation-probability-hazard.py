import os
import sys
import re
import pickle
import time
import torch
import numpy as np
import pandas as pd
from huggingface_hub import login
from getpass import getpass
from tqdm.autonotebook import tqdm

from resources.preprocessing import PromptLoader
from resources.modelling import ChatGenerationPipeline

torch.manual_seed(42)
if os.path.exists('.huggingface.token'):
    with open('.huggingface.token', 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

MAX_SEQ_LEN = 512
MAX_GEN_LEN = 128
MODEL_NAME  = sys.argv[1]
SAVE_PREFIX = '' if len(sys.argv) < 3 else sys.argv[2]

#====================================================================================================#
# Prepare:                                                                                           #
#====================================================================================================#

# Load Pipeline:
pipe = ChatGenerationPipeline.from_pretrained(
    MODEL_NAME,
    max_seq_len = MAX_SEQ_LEN,
    max_gen_len = MAX_GEN_LEN,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
 
# Load data:
data_test   = pd.read_csv('data/food incidents - hazard/incidents_sample.csv', index_col=0)[['title', 'hazard-category', 'hazard-raw']].fillna('')
labels_test = data_test['hazard-category'].unique().tolist()
tokens_test = [pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(l)['input_ids'][1:]) for l in labels_test]

#====================================================================================================#
# Prompt:                                                                                            #
#====================================================================================================#

loader = PromptLoader(
    prefix    = "What is the reason for the recall of the food product in the following announcement?\n\n",
    suffix    = "\n\nAssign one of the following labels: \"biological\", \"allergens\", \"chemical\", \"foreign bodies\", \"organoleptic aspects\", or \"fraud\". Make sure to answer only with the label or \"none\" if none of them applies.",
    separator = ""
)
prefix_size = pipe.countTokens(loader.prefix, sot=True)
suffix_size = pipe.countTokens(loader.suffix, eot=True)

path = f'results/food incidents - hazard/{SAVE_PREFIX}{MODEL_NAME}_probability.pkl'
os.makedirs(os.path.dirname(path), exist_ok=True)

results = []
rex = re.compile('"(.+)"', re.DOTALL)
for step, s in enumerate(tqdm(data_test.values)):
    #================================================================================================#
    # Classification Task:                                                                           #
    #================================================================================================#

    # create prompt:
    p0, label, spans = loader.createPrompt(tuple(s))

    # generation:
    chat, input_ids, output_ids = pipe.generate(p0, output_attentions=True, output_hidden_states=True, compute_grads=tokens_test)

    result = {
        'chat': chat,
        'tokens': pipe.tokenizer.convert_ids_to_tokens(output_ids[0]),
        'sample': {
            'text': pipe.tokenizer.decode(input_ids[0,prefix_size:-suffix_size]),
            'start': prefix_size,
            'end': input_ids.shape[1] - suffix_size,
        },
        'label': {
            'text': label.lower(),
            'tokens': tokens_test[labels_test.index(label)],
        },
        'prediction': {
            'text': chat[1][1].lower().split('\n')[0].strip('"*. '),
            'index': input_ids.shape[1]
        },
        'probabilities': {
            'precise': {},
            'approximated': {}
        }
    }

    #================================================================================================#
    # Label Probabilities:                                                                           #
    #================================================================================================#

    order = np.random.permutation(len(labels_test))

    # approximated:
    for i in order:
        label, token = labels_test[i], tokens_test[i]

        t = time.time_ns()
        p = pipe.getOutputProbability(token, precise=False).detach().to(device='cpu', dtype=float).numpy()
        t = time.time_ns() - t

        result['probabilities']['approximated'][label] = {'p': p, 'dt': t}

    # precise:
    for i in order:
        label, token = labels_test[i], tokens_test[i]

        t = time.time_ns()
        p = pipe.getOutputProbability(token, precise=True).detach().to(device='cpu', dtype=float).numpy()
        t = time.time_ns() - t

        result['probabilities']['precise'][label] = {'p': p, 'dt': t}

    # append to results:
    results.append(result)

    # save:
    if step % 5 == 0:
        with open(path, 'wb') as file:
            pickle.dump(results, file)

# save:
with open(path, 'wb') as file:
    pickle.dump(results, file)