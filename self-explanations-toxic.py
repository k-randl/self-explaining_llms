import os
import sys
import re
import pickle
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

MAX_SEQ_LEN   = 512
MAX_GEN_LEN   = 128
MAX_LABEL_LEN = 10
MODEL_NAME    = sys.argv[1]

PRECISE_PROBS    = True
PRECISE_SALIENCY = True
SAVE_PREFIX      = ''
for i, v in enumerate(sys.argv[2:]):
    if   v == '-p': PRECISE_PROBS = False
    elif v == '-s': PRECISE_SALIENCY = False
    elif i == 0:    SAVE_PREFIX = v
    else: raise ValueError(f'Unknown command line parameter "{v}".')

if not PRECISE_PROBS: print('Using approximated label probabilities.')
if not PRECISE_SALIENCY: print('Using approximated token saliencies.')

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
data_test   = pd.read_csv('data/toxic spans/sample.csv', index_col=0)[['comment_text', 'label', 'spans']]
labels_test = ['insult', 'identity based attack', 'profane/obscene', 'threat']
tokens_test = [pipe.stripBOS(pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(l)['input_ids'])) for l in labels_test]

#====================================================================================================#
# Prompt:                                                                                            #
#====================================================================================================#

loader = PromptLoader(
    prefix    = "What kind of toxic language is present in the following post?\n\n",
    suffix    = "\n\nAssign one of the following labels: \"insult\", \"identity based attack\", \"profane/obscene\", or \"threat\". Make sure to answer only with the label or \"none\" if none of them applies.",
    separator = ""
)
prefix_size = pipe.countTokens(loader.prefix, sot=True)
suffix_size = pipe.countTokens(loader.suffix, eot=True)

path = f'results/toxic spans/{SAVE_PREFIX}{MODEL_NAME}.pkl'
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
    chat, input_ids, output_ids = pipe.generate(p0, output_attentions=True, output_hidden_states=True, max_new_tokens=MAX_LABEL_LEN, compute_grads=tokens_test)

    result = {
        'chat': chat,
        'tokens': pipe.tokenizer.convert_ids_to_tokens(output_ids[0]),
#        'hidden_states': pipe.model.hiddenStates.detach().to(device='cpu', dtype=float).numpy(),
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
        'perturbation': {},
        'spans': {},
    }

    result['AGrad'] = pipe.aGrad(precise=PRECISE_SALIENCY)
    result['GradIn'] = pipe.gradIn(precise=PRECISE_SALIENCY, window=slice(result['sample']['start'],result['sample']['end']), epsilon=1e-2)
    result['GradH'] = pipe.gradH(precise=PRECISE_SALIENCY, window=slice(result['sample']['start'],result['sample']['end']), epsilon=1e-2)
    result['IGrad'], result['IGrad-tokens'] = pipe.iGrad(max_tokens=2048, precise=PRECISE_SALIENCY)
    result['Shap'] = pipe.shap(max_evals=500, fixed_prefix=loader.prefix, fixed_suffix=loader.suffix, max_new_tokens=MAX_LABEL_LEN)

    # generate counterfactual label:
    label_probs = [(l, pipe.getOutputProbability(t, precise=PRECISE_PROBS).detach().to(device='cpu', dtype=float).numpy()) for l, t in zip(labels_test, tokens_test) if l.lower() != result['prediction']['text']]
    label_probs.sort(key=lambda x: x[1], reverse=True)

    cf_label = label_probs[0][0]


    #================================================================================================#
    # Analytic Explanations:                                                                         #
    #================================================================================================#

    # get prediction boundaries:
    y_start = result['prediction']['index']

    pt = PerturbationTester(
        input_ids, result['sample']['start'], result['sample']['end'], pipe.mask_token_id,
        pipe.model, pipe.tokenizer,
        max_new_tokens=MAX_LABEL_LEN
    )

    try: prediction_id = labels_test.index(result['prediction']['text'])
    except ValueError: prediction_id = None

    counterfactual_id = labels_test.index(cf_label)

    # attention based explanations:
    #importance = result['AGrad'].mean(axis=0)[prediction_id]
    #importance = result['AGrad'].mean(axis=(0, 1))
    if prediction_id is None: importance = result['AGrad'].mean(axis=(0, 1))
    else:                     importance = result['AGrad'].mean(axis=0)[prediction_id]
    result['perturbation']['AGrad'] = pt.test(importance, double_sided=True)
    result['AGrad'] = result['AGrad'].detach().to(device='cpu', dtype=float).numpy()

    # gradient based explanations:
    #importance = result['GradIn'][prediction_id]
    #importance = result['GradIn'].mean(axis=0)
    if prediction_id is None: importance = result['GradIn'].mean(axis=0)
    else:                     importance = result['GradIn'][prediction_id]
    result['perturbation']['GradIn'] = pt.test(importance, double_sided=True)
    result['GradIn'] = result['GradIn'].detach().to(device='cpu', dtype=float).numpy()

    # gradient based explanations:
    #importance = result['GradH'][prediction_id]
    #importance = result['GradH'].mean(axis=0)
    if prediction_id is None: importance = result['GradH'].mean(axis=0)
    else:                     importance = result['GradH'][prediction_id]
    result['perturbation']['GradH'] = pt.test(importance, double_sided=True)
    result['GradH'] = result['GradH'].detach().to(device='cpu', dtype=float).numpy()

    #importance = result['IGrad'].mean(axis=-1)[prediction_id]
    #importance = result['IGrad'].mean(axis=(0, -1))
    importance = result['IGrad'].mean(axis=-1)[counterfactual_id]
    result['perturbation']['IGrad'] = pt.test(importance, double_sided=True)
    result['IGrad'] = result['IGrad'].detach().to(device='cpu', dtype=float).numpy()

    # model agnostic explanations:
    if prediction_id is None: importance = result['Shap'].mean(axis=0)
    else:                     importance = result['Shap'][prediction_id]
    result['perturbation']['Shap'] = pt.test(importance, double_sided=True)
    result['Shap'] = result['Shap'].detach().to(device='cpu', dtype=float).numpy()


    #================================================================================================#
    # Prompting-Based Explanations:                                                                  #
    #================================================================================================#

    # self assessment:
    chat, _, _ = pipe.generate("What is the most important phrase of the post influencing your assessment? Provide only the phrase as a string.", output_ids)
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
    chat, _, _ = pipe.generate(f"Provide a version of the post that would alter your assessment to \"{cf_label}\" while changing as few words in the original post as possible. Make sure to only answer with the changed post.", output_ids)
    result['chat'].extend(chat[2:])

    cf = rex.findall(chat[3][1])
    cf = chat[3][1] if len(cf) == 0 else cf[0]
    p1, _, _ = loader.createPrompt((cf, False, None))
    chat, input_ids_cf, _ = pipe.generate(p1, output_hidden_states=True, max_new_tokens=MAX_LABEL_LEN)
    similarity, spans = similarityTest(input_ids, input_ids_cf)
    result['counterfactual'] = {
        'text'         : cf,
        'target_label' : cf_label,
        'prediction'   : chat[1][1].lower(),
        'similarity'   : similarity,
#        'hidden_states': pipe.model.hiddenStates.detach().to(device='cpu', dtype=float).numpy(),
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