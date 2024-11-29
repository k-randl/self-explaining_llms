import os
import sys
import re
import pickle
import random
import torch
import pandas as pd
from huggingface_hub import login
from getpass import getpass
from tqdm.autonotebook import tqdm

from resources.preprocessing import PromptLoader
from resources.modelling import ChatGenerationPipeline
from resources.testing import PerturbationTester, similarityTest, findSpan

torch.manual_seed(42)
if os.path.exists('.huggingface.token'):
    with open('.huggingface.token', 'r') as file:
        login(token=file.read())

else: login(token=getpass(prompt='Huggingface login  token: '))

MAX_SEQ_LEN = 512
MAX_GEN_LEN = 128
MODEL_NAME  = sys.argv[1]

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

    # generation:
    chat, input_ids, output_ids = pipe.generate(p0, output_attentions=True, output_hidden_states=True, compute_grads=tokens_test)

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
            'tokens': tokens_test[labels_test.index(label)],
        },
        'prediction': {
            'text': chat[1][1].lower().split('\n\n')[0].strip('"*.'),
            'index': input_ids.shape[1]
        },
        'perturbation': {},
        'spans': {},
        'AGrad': pipe.aGrad(),
        'Grad': pipe.grad(),
        'GradIn': pipe.gradIn(),
        'IGrad': pipe.iGrad(),
    }

    # generate counterfactual label:
    label_probs = [(l, pipe.approximateOutputProbability(t)) for l, t in zip(labels_test, tokens_test) if l.lower() != result['prediction']['text']]
    label_probs.sort(key=lambda x: x[1], reverse=True)

    cf_label = label_probs[0][0]


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
    importance = result['AGrad'].mean(axis=(1,))[labels_test.index(result['prediction']['text'])]
    result['perturbation']['AGrad'] = pt.test(importance, double_sided=True)
    result['AGrad'] = result['AGrad'].detach().cpu().numpy()

    # gradient based explanations:
    importance = result['GradIn'].mean(axis=(-1,))[labels_test.index(result['prediction']['text'])]
    result['perturbation']['GradIn'] = pt.test(importance, double_sided=True)
    result['GradIn'] = result['GradIn'].detach().cpu().numpy()

    importance = result['IGrad'].mean(axis=(-1,))[labels_test.index(cf_label)]
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
    chat, _, _ = pipe.generate(f"Provide a version of the announcement that would alter your assessment to \"{cf_label}\" while changing as few words in the original announcement as possible.", output_ids)
    result['chat'].extend(chat[2:])

    cf = rex.findall(chat[3][1])
    cf = chat[3][1] if len(cf) == 0 else cf[0]
    p1, _, _ = loader.createPrompt((cf, False, None))
    chat, input_ids_cf, _ = pipe.generate(p1, output_hidden_states=True)
    similarity, spans = similarityTest(input_ids, input_ids_cf)
    result['counterfactual'] = {
        'text'         : cf,
        'target_label' : cf_label,
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