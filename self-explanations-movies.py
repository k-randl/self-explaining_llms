import os
import sys
import re
import pickle
import random
import torch
from huggingface_hub import login
from getpass import getpass
from tqdm.autonotebook import tqdm

from resources.preprocessing import PromptLoaderIMDB
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
with open('data/movies/val_sample.jsonl', 'r') as file:
    data_test = file.read().split('\n')
labels_test = ['negative', 'positive']
tokens_test = [pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(l)['input_ids'][1:]) for l in labels_test]

#====================================================================================================#
# Prompt:                                                                                            #
#====================================================================================================#

loader = PromptLoaderIMDB(
    prefix    = "What is the sentiment of the following review?\n\n",
    postfix   = "\n\nAssign one of the following labels: \"negative\" or \"positive\". Make sure to answer only with the label.",
    separator = ""
)
prefix_size  = pipe.countTokens(loader.prefix, sot=True)
postfix_size = pipe.countTokens(loader.postfix, eot=True)

path = f'results/movies/{MODEL_NAME}.pkl'
os.makedirs(os.path.dirname(path), exist_ok=True)

results = []
rex = re.compile('"(.+)"', re.DOTALL)
for step, s in enumerate(tqdm(data_test[:-1])):
    #================================================================================================#
    # Classification Task:                                                                           #
    #================================================================================================#

    # create prompt:
    p0, label, spans = loader.createPrompt(s)

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
            'tokens': tokens_test[labels_test.index(label.lower())],
        },
        'prediction': {
            'text': chat[1][1][:8].lower(),
            'index': input_ids.shape[1]
        },
        'perturbation': {},
        'spans': {},
        'AGrad': pipe.aGrad(),
        'Grad': pipe.grad(),
        'GradIn': pipe.gradIn(),
        'IGrad': pipe.iGrad(),
    }


    #================================================================================================#
    # Analytic Explanations:                                                                         #
    #================================================================================================#

    # get prediction boundaries:
    y_start = result['prediction']['index']
    y       = int(result['prediction']['text'] == 'positive')

    # generate counterfactual label:
    cf_label = labels_test[1-y]

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
    if len(spans) <= 1: p1 = "What is the most important phrase of the review influencing your assessment? Provide only the phrase as a string."
    else:               p1 = f"What are the {len(spans):d} most important phrases of the review influencing your assessment? Provide a list of strings with one phrase per line."
    chat, _, _ = pipe.generate(p1, output_ids)
    result['chat'].extend(chat[2:])

    result['spans']['human'] = [findSpan(pipe.getPrompt(p0, bos=False), loader.decode(s['text']).replace('"', '').lower(), pipe.tokenizer) for s in spans]
    result['spans']['extractive'] = [findSpan(pipe.getPrompt(p0, bos=False), s.strip('"-•* '), pipe.tokenizer) for s in chat[3][1].lower().split('\n')]

    # counterfactual:
    chat, _, _ = pipe.generate(f"Provide a version of the announcement that would alter your assessment to \"{cf_label}\" while changing as few words in the original announcement as possible.", output_ids)
    result['chat'].extend(chat[2:])

    cf = rex.findall(chat[3][1])
    cf = chat[3][1] if len(cf) == 0 else cf[0]
    p2, _, _ = loader.createPrompt((cf, False, None))
    chat, input_ids_cf, _ = pipe.generate(p2, output_hidden_states=True)
    similarity, spans = similarityTest(input_ids, input_ids_cf)
    result['counterfactual'] = {
        'text'         : cf,
        'target_label' : cf_label,
        'prediction'   : chat[1][1][:8].lower(),
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