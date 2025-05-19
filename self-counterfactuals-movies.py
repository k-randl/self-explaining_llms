import os
import sys
import re
import time
import pickle
import torch
import pandas as pd
from huggingface_hub import login
from getpass import getpass
from tqdm.autonotebook import tqdm

from resources.preprocessing import PromptLoaderIMDB
from resources.modelling import ChatGenerationPipeline, no_explain
from resources.testing import similarityTest

torch.manual_seed(42)
if os.path.exists('.huggingface.token'):
    with open('.huggingface.token', 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

MAX_SEQ_LEN   = 512
MAX_GEN_LEN   = 128
MAX_LABEL_LEN = 8
MODEL_NAME    = sys.argv[1]
NUM_SAMPLES   = int(sys.argv[2])
TEMPERATURE   = float(sys.argv[3])

PRECISE_PROBS = True
SAVE_PREFIX   = ''
for i, v in enumerate(sys.argv[4:]):
    if   v == '-p': PRECISE_PROBS = False
    elif i == 0:    SAVE_PREFIX = v
    else: raise ValueError(f'Unknown command line parameter "{v}".')

if not PRECISE_PROBS: print('Using approximated label probabilities.')

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
with open('data/movies/val_sample.jsonl', 'r') as file:
    data_test = file.read().split('\n')
labels_test = ['negative', 'positive']
tokens_test = [pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(l)['input_ids'][1:]) for l in labels_test]

#====================================================================================================#
# Prompt:                                                                                            #
#====================================================================================================#

loader = PromptLoaderIMDB(
    prefix    = "What is the sentiment of the following review?\n\n",
    suffix    = "\n\nAssign one of the following labels: \"negative\" or \"positive\". Make sure to answer only with the label or \"none\" if none of them applies.",
    separator = ""
)
prefix_size = pipe.countTokens(loader.prefix, sot=True)
suffix_size = pipe.countTokens(loader.suffix, eot=True)

path = f'results/movies/{SAVE_PREFIX}{MODEL_NAME}-self-counterfactuals.pkl'
os.makedirs(os.path.dirname(path), exist_ok=True)

results = []
rex = re.compile('"(.+)"', re.DOTALL)
with no_explain(pipe, reset_on_generate=True):
    for step, s in enumerate(tqdm(data_test[:-1])):
        #================================================================================================#
        # Classification Task:                                                                           #
        #================================================================================================#

        # create prompt:
        prompt, label, spans = loader.createPrompt(s)

        # generation:
        chat, input_ids, output_ids = pipe.generate(prompt, output_attentions=False, output_hidden_states=False, max_new_tokens=MAX_LABEL_LEN, compute_grads=tokens_test)

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
                'tokens': tokens_test[labels_test.index(label.lower())],
            },
            'prediction': {
                'text': chat[1][1].lower().split('\n')[0].strip('"*. '),
                'index': input_ids.shape[1]
            },
            'counterfactuals': {},
        }

        #================================================================================================#
        # Counterfactual Sampling:                                                                       #
        #================================================================================================#

        # generate counterfactual label:
        label_probs = [(l, pipe.getOutputProbability(t, precise=PRECISE_PROBS).detach().to(device='cpu', dtype=float).numpy()) for l, t in zip(labels_test, tokens_test) if l.lower() != result['prediction']['text']]
        label_probs.sort(key=lambda x: x[1], reverse=True)

        
        for l, p in label_probs:
            result['counterfactuals'][l] = {'probability': float(p), 'samples': []}

            # get tokens in the current label:
            tokens_l = tokens_test[labels_test.index(l)]

            # try to sample at least NUM_SAMPLES valid counterfactuals
            # in maximum NUM_SAMPLES*2 trys:
            n_valid = 0
            for _ in range(NUM_SAMPLES*2):
                t = time.time_ns()

                chat, _, _ = pipe.generate(f"Provide a version of the review that would alter your assessment to \"{l}\" while changing as few words in the original review as possible. Make sure to only answer with the changed review.", output_ids, do_sample=True, temperature=TEMPERATURE)

                cf = rex.findall(chat[3][1])
                cf = chat[3][1] if len(cf) == 0 else cf[0]

                # test counterfactual candidate:
                prompt, _, _ = loader.createPrompt((cf, False, None))
                chat, input_ids_cf, _ = pipe.generate(prompt, output_attentions=False, output_hidden_states=False, max_new_tokens=MAX_LABEL_LEN, compute_grads=tokens_test)
                prediction = chat[1][1].lower()

                t = time.time_ns() - t

                # calculate probability of candidate to be predicted to `l`:
                probability = pipe.getOutputProbability(tokens_l, precise=PRECISE_PROBS).detach().to(device='cpu', dtype=float).numpy()

                # calculate candidate similarity:
                similarity, spans = similarityTest(input_ids, input_ids_cf)

                result['counterfactuals'][l]['samples'].append({
                    'text'         : cf,
                    'prediction'   : prediction,
                    'probability'  : probability,
                    'similarity'   : similarity,
                    'spans'        : spans,
                    'dt'           : t
                })

                n_valid += int(prediction == l)
                if n_valid >= NUM_SAMPLES:
                    break

        # append to results:
        results.append(result)

        # save:
        if step % 5 == 0:
            with open(path, 'wb') as file:
                pickle.dump(results, file)

# save:
with open(path, 'wb') as file:
    pickle.dump(results, file)