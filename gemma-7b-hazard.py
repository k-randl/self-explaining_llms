import os
import re
import pickle
import torch
import pandas as pd
from huggingface_hub import login
from getpass import getpass
from transformers import GenerationConfig
from tqdm.autonotebook import tqdm

from resources.preprocessing import PromptLoader
from resources.modelling import GemmaChatGenerationPipeline
from resources.testing import PerturbationTester, similarityTest, findSpan

torch.manual_seed(42)
login(token=getpass(prompt='Huggingface login  token: '))

MAX_SEQ_LEN = 512
MAX_GEN_LEN = 128
MODEL_NAME  = "google/gemma-1.1-7b-it"

#====================================================================================================#
# Prepare:                                                                                           #
#====================================================================================================#

# Load data:
data_test = pd.read_csv('data/food incidents - hazard/incidents_sample.csv', index_col=0)[['title', 'hazard-category', 'hazard-raw']].fillna('')

# Load Pipeline:
pipe = GemmaChatGenerationPipeline.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)
pipe.model.generation_config = GenerationConfig(
    bos_token_id = pipe.tokenizer.bos_token_id, #2
    eos_token_id = pipe.tokenizer.eos_token_id, #1
    pad_token_id = pipe.tokenizer.pad_token_id, #0
    
    max_length = MAX_SEQ_LEN,
    max_new_tokens = MAX_GEN_LEN,
)
pipe.model.config

#====================================================================================================#
# Prompt:                                                                                            #
#====================================================================================================#

loader = PromptLoader(
    prefix    = "What is the reason for the recall of the food product in the following announcement?\n\n",
    postfix   = "\n\nAssign one of the following labels: \"biological\", \"allergens\", \"chemical\", \"foreign bodies\", \"organoleptic aspects\", or \"fraud\". Make sure to answer only with the label.",
    separator = ""
)
prefix_size  = pipe.countTokens(loader.prefix, sot=True)
postfix_size = pipe.countTokens(loader.postfix, eot=True)

path = f'results/food incidents - hazard/{MODEL_NAME}.pkl'
os.makedirs(os.path.dirname(path), exist_ok=True)

results = []
rex = re.compile('"(.+)"', re.DOTALL)
for step, s in enumerate(tqdm(data_test.values)):
    #================================================================================================#
    # Classification Task:                                                                           #
    #================================================================================================#

    # create prompt:
    p0, label, spans = loader.createPrompt(tuple(s))
    outputs = pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(label)['input_ids'][1:])

    # generation:
    chat, input_ids, output_ids = pipe.generate(p0, output_attentions=True, output_hidden_states=True, compute_grads=set(outputs))
    
    result = {
        'chat': chat,
        'tokens': pipe.tokenizer.convert_ids_to_tokens(output_ids[0]),
        'hidden_states': pipe.model.hiddenStates.detach().cpu().numpy(),
        'sample': {
            'text': pipe.tokenizer.decode(input_ids[0,prefix_size:-postfix_size]),
            'start': prefix_size,
            'end': input_ids.shape[1] - postfix_size,
        },
        'label': {
            'text': label.lower(),
            'tokens': outputs,
        },
        'prediction': {
            'text': chat[1][1].lower().split('\n\n')[0].strip('"*.'),
            'index': input_ids.shape[1]
        },
        'perturbation': {},
        'spans': {},
        'AGrad': pipe.model.aGrad(outputs=outputs),
        'Grad': pipe.model.getHiddenStateGradients(outputs=outputs),
        'GradIn': pipe.model.GradIn(outputs=outputs),
        'IGrad': pipe.model.iGrad(outputs=outputs),
    }


    #================================================================================================#
    # Analytic Explanations:                                                                         #
    #================================================================================================#
    
    # get prediction boundaries:
    y_start = result['prediction']['index']

    pt = PerturbationTester(
        input_ids, result['sample']['start'], result['sample']['end'], pipe.mask_token_id,
        pipe.model, pipe.tokenizer
    )

    # attention based explanations:
    importance = result['AGrad'][0, :, :, y_start-1:, :y_start].cumsum(axis=0).mean(axis=(0,1,2))
    result['perturbation']['AGrad'] = pt.test(importance, double_sided=True)
    result['AGrad'] = result['AGrad'].detach().cpu().numpy()

    # gradient based explanations:
    importance = result['GradIn'][0, :, :y_start, :].cumsum(axis=0).mean(axis=(0,-1))
    result['perturbation']['GradIn'] = pt.test(importance, double_sided=True)
    result['GradIn'] = result['GradIn'].detach().cpu().numpy()

    importance = result['IGrad'][0, :y_start, :, :].cumsum(axis=-1).mean(axis=(-1,-2))
    result['perturbation']['IGrad'] = pt.test(importance, double_sided=True)
    result['IGrad'] = result['IGrad'].detach().cpu().numpy()


    #================================================================================================#
    # Prompting-Based Explanations:                                                                  #
    #================================================================================================#

    # self assessment:
    chat, _, _ = pipe.generate("What is the most important phrase of the announcement influencing your assessment? Provide only the phrase as a string.", output_ids)
    result['chat'].extend(chat[2:])

    result['spans']['human'] = []

    i, j = findSpan(pipe.getPrompt(p0, bos=False), label.replace('"', '').lower(), pipe.tokenizer)
    if i >= result['sample']['start'] and j <= result['sample']['end']:
        result['spans']['human'].append((i,j))

    i, j = findSpan(pipe.getPrompt(p0, bos=False), spans.replace('"', '').lower(), pipe.tokenizer)
    if i >= result['sample']['start'] and j <= result['sample']['end']:
        result['spans']['human'].append((i,j))

    result['spans']['extractive'] = [findSpan(pipe.getPrompt(p0, bos=False), s.strip('"- '), pipe.tokenizer) for s in chat[3][1].lower().split('\n')]

    # counterfactual:
    chat, _, _ = pipe.generate("Provide a version of the announcement that would alter your assessment while changing as few words in the original announcement as possible.", output_ids)
    result['chat'].extend(chat[2:])

    cf = rex.findall(chat[3][1])
    cf = chat[3][1] if len(cf) == 0 else cf[0]
    p1, _, _ = loader.createPrompt((cf, False, None))
    chat, input_ids_cf, _ = pipe.generate(p1, output_hidden_states=True)
    similarity, spans = similarityTest(input_ids, input_ids_cf)
    result['counterfactual'] = {
        'text'         : cf,
        'prediction'   : chat[1][1].lower(),
        'similarity'   : similarity,
        'hidden_states': pipe.model.hiddenStates.detach().cpu().numpy(),
    }
    result['spans']['counterfactual'] = spans

    # perturbation tests:
    for key in result['spans']:
        result['perturbation'][key] = pt.testSpans(result['spans'][key])

    # append to results:
    results.append(result)

    # save:
    if step % 5 == 0:
        with open(path, 'wb') as file:
            pickle.dump(results, file)

# save:
with open(path, 'wb') as file:
    pickle.dump(results, file)