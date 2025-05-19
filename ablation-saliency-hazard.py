import os
import sys
import re
import pickle
import random
import time
import torch
import pandas as pd
from huggingface_hub import login
from getpass import getpass
from tqdm.autonotebook import tqdm

from resources.preprocessing import PromptLoader
from resources.modelling import ChatGenerationPipeline, filter_noise
from resources.testing import PerturbationTester

torch.manual_seed(42)
if os.path.exists('.huggingface.token'):
    with open('.huggingface.token', 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

MAX_SEQ_LEN = 512
MAX_GEN_LEN = 10
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
labels_test = ['allergens', 'biological', 'foreign bodies', 'chemical', 'organoleptic aspects', 'fraud']
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

path = f'results/food incidents - hazard/{SAVE_PREFIX}{MODEL_NAME}_saliency.pkl'
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
    }

    try: prediction_id = labels_test.index(result['prediction']['text'])
    except ValueError: prediction_id = None

    result['perturbation'] = {}
    result['dt'] = {}

    #================================================================================================#
    # AGrad:                                                                                         #
    #================================================================================================#

    # approximated:
    t = time.time_ns()
    result['AGrad-approximated'] = pipe.aGrad(precise=False)
    result['dt']['AGrad-approximated'] = time.time_ns() - t

    # precise:
    for aggregation in ['saliency', 'cumsum', 'equal']:
        t = time.time_ns()
        result[f'AGrad-{aggregation}'] = pipe.aGrad(precise=True, aggregation=aggregation)
        result['dt'][f'AGrad-{aggregation}'] = time.time_ns() - t

    #================================================================================================#
    # GradH:                                                                                          #
    #================================================================================================#

    # approximated:
    t = time.time_ns()
    result['GradH-approximated'] = pipe.gradH(precise=False, normalize=False)
    result['dt']['GradH-approximated'] = time.time_ns() - t

    # precise:
    for aggregation in ['saliency', 'cumsum', 'equal']:
        t = time.time_ns()
        result[f'GradH-{aggregation}'] = pipe.gradH(precise=True, normalize=False, aggregation=aggregation)
        result['dt'][f'GradH-{aggregation}'] = time.time_ns() - t

    #================================================================================================#
    # GradIn:                                                                                        #
    #================================================================================================#

    # approximated:
    t = time.time_ns()
    result['GradIn-approximated'] = pipe.gradIn(precise=False, normalize=False, caching=False)
    result['dt']['GradIn-approximated'] = time.time_ns() - t

    # precise:
    for aggregation in ['saliency', 'cumsum', 'equal']:
        t = time.time_ns()
        result[f'GradIn-{aggregation}'] = pipe.gradIn(precise=True, normalize=False, caching=False, aggregation=aggregation)
        result['dt'][f'GradIn-{aggregation}'] = time.time_ns() - t

    #================================================================================================#
    # Shap:                                                                                          #
    #================================================================================================#

    # precise:
    #t = time.time_ns()
    #result['Shap'] = pipe.shap(max_evals=500, fixed_prefix=loader.prefix, fixed_suffix=loader.suffix)
    #result['dt']['Shap'] = time.time_ns() - t

    #================================================================================================#
    # Perturbations:                                                                                 #
    #================================================================================================#

    pt = PerturbationTester(
        input_ids, result['sample']['start'], result['sample']['end'], pipe.mask_token_id,
        pipe.model, pipe.tokenizer
    )

    def test_raw(importance:torch.FloatTensor):
        if prediction_id is None: return pt.test(torch.zeros_like(importance[0]))
        else:                     return pt.test(importance[prediction_id])

    def test_filtered(importance:torch.FloatTensor):
        filtered = filter_noise(importance, window=slice(result['sample']['start'], result['sample']['end']), epsilon=1e-2)
        return test_raw(filtered)

    for aggregation in ['approximated', 'saliency', 'cumsum', 'equal']:
        result['perturbation'][f'AGrad-{aggregation} (raw)']  = test_raw(result[f'AGrad-{aggregation}'].mean(axis=0))
        result['perturbation'][f'GradH-{aggregation} (raw)']  = test_raw(result[f'GradH-{aggregation}'])
        result['perturbation'][f'GradIn-{aggregation} (raw)'] = test_raw(result[f'GradIn-{aggregation}'])

        result['perturbation'][f'AGrad-{aggregation} (filtered)']  = test_filtered(result[f'AGrad-{aggregation}'].mean(axis=0))
        result['perturbation'][f'GradH-{aggregation} (filtered)']  = test_filtered(result[f'GradH-{aggregation}'])
        result['perturbation'][f'GradIn-{aggregation} (filtered)'] = test_filtered(result[f'GradIn-{aggregation}'])

    #result['perturbation']['Shap'] = test_raw(result['Shap'])

    #================================================================================================#
    # Clean Up:                                                                                      #
    #================================================================================================#
    
    for method in ['AGrad', 'GradH', 'GradIn']:
        for aggregation in ['approximated', 'saliency', 'cumsum', 'equal']:
            result[f'{method}-{aggregation}'] = result[f'{method}-{aggregation}'].detach().to(device='cpu', dtype=float).numpy()

    #result['Shap'] = result['Shap'].detach().to(device='cpu', dtype=float).numpy()

    # append to results:
    results.append(result)

    # save:
    if step % 5 == 0:
        with open(path, 'wb') as file:
            pickle.dump(results, file)

# save:
with open(path, 'wb') as file:
    pickle.dump(results, file)