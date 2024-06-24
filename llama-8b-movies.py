import os
import re
import pickle
import torch
from huggingface_hub import login
from getpass import getpass
from transformers import GenerationConfig
from tqdm.autonotebook import tqdm

from resources.preprocessing import PromptLoaderIMDB
from resources.modelling import ChatGenerationPipeline
from resources.testing import PerturbationTester, similarityTest, findSpan

torch.manual_seed(42)
login(token=getpass(prompt='Huggingface login  token: '))

MAX_SEQ_LEN = 512
MAX_GEN_LEN = 128
MODEL_NAME  = "meta-llama/Meta-Llama-3-8B-Instruct"

#====================================================================================================#
# Prepare:                                                                                           #
#====================================================================================================#

# Load data:
with open('data/movies/val_sample.jsonl', 'r') as file:
    data_test = file.read().split('\n')

# Load Pipeline:
pipe = ChatGenerationPipeline.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)
pipe.model.generation_config = GenerationConfig(
    bos_token_id = pipe.tokenizer.convert_tokens_to_ids('<|begin_of_text|>'),
    eos_token_id = pipe.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
    pad_token_id = pipe.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
    
    max_length = MAX_SEQ_LEN,
    max_new_tokens = MAX_GEN_LEN,
)
pipe.model.config

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
    outputs = ['Negative', 'Positive']

    # generation:
    chat, input_ids, output_ids = pipe.generate(p0, output_attentions=True, output_hidden_states=True, compute_grads=outputs)
    
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
            'tokens': [label],
        },
        'prediction': {
            'text': chat[1][1][:8].lower(),
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
    y       = int(result['prediction']['text'] == 'positive')

    pt = PerturbationTester(
        input_ids, result['sample']['start'], result['sample']['end'], pipe.mask_token_id,
        pipe.model, pipe.tokenizer,
        pad_token_id = pipe.tokenizer.convert_tokens_to_ids('<|eot_id|>')
    )

    # attention based explanations:
    importance = result['AGrad'][0, y, :, y_start-1, :y_start].mean(axis=0)
    result['perturbation']['AGrad'] = pt.test(importance, double_sided=True)
    result['AGrad'] = result['AGrad'].detach().cpu().numpy()

    # gradient based explanations:
    importance = result['GradIn'][0, y, :y_start, :].mean(axis=-1)
    result['perturbation']['GradIn'] = pt.test(importance, double_sided=True)
    result['GradIn'] = result['GradIn'].detach().cpu().numpy()

    importance = result['IGrad'][0, :y_start, :, y].mean(axis=1)
    result['perturbation']['IGrad'] = pt.test(importance, double_sided=True)
    result['IGrad'] = result['IGrad'].detach().cpu().numpy()


    #================================================================================================#
    # Prompting-Based Explanations:                                                                  #
    #================================================================================================#

    # self assessment:
    if len(spans) <= 1: p1 = "What is the most important phrase of the review influencing your assessment? Provide only the phrase as a string."
    else:               p1 = f"What are the {len(spans):d} most important phrases of the review influencing your assessment? Provide only a list of strings with one phrase per line."
    chat, _, _ = pipe.generate(p1, output_ids)
    result['chat'].extend(chat[2:])

    result['spans']['human'] = [findSpan(pipe.getPrompt(p0, bos=False), loader.decode(s['text']).replace('"', '').lower(), pipe.tokenizer) for s in spans]
    result['spans']['extractive'] = [findSpan(pipe.getPrompt(p0, bos=False), s.strip('"-•* '), pipe.tokenizer) for s in chat[3][1].lower().split('\n')]

    # counterfactual:
    chat, _, _ = pipe.generate("Provide a version of the review that would flip your assessment while changing as few words in the original review as possible. Make sure to answer with only the new version.", output_ids)
    result['chat'].extend(chat[2:])

    cf = rex.findall(chat[3][1])
    cf = chat[3][1] if len(cf) == 0 else cf[0]
    p2, _, _ = loader.createPrompt((cf, False, None))
    chat, input_ids_cf, _ = pipe.generate(p2, output_hidden_states=True)
    similarity, spans = similarityTest(input_ids, input_ids_cf)
    result['counterfactual'] = {
        'text'         : cf,
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