import json
import random

from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import Any, List, Tuple, Union, Optional

#=======================================================================#
# Dataloader:                                                           #
#=======================================================================#

class PromptLoader():
    def __init__(self, prefix:str, suffix:str=' => ', separator:str='\n\n', seed:Optional[int]=None) -> None:
        super().__init__()
        if isinstance(seed, int): random.seed(seed)

        # static prefix and suffix of prompts:
        self.prefix    = prefix
        self.suffix    = suffix
        self.separator = separator

        # detokenizer for imdb dataset.
        self._detokenizer = TreebankWordDetokenizer()

    def _detokenize(self, tokens:List[str]):
        # start sentences with capital letter:
        tokens[0] = tokens[0][0].upper() + tokens[0][1:]
        for i in range(len(tokens)-1):
            if tokens[i] in ('.', '!', '?'):
                tokens[i+1] = tokens[i+1][0].upper() + tokens[i+1][1:]

        # detokenize:
        txt = self._detokenizer.detokenize(tokens)

        # replace certain chars:
        txt = txt.replace(' -', '-')
        txt = txt.replace('- ', '-')
        txt = txt.replace(' i ', ' I ')
        txt = txt.replace(' .', '.')
        txt = txt.replace(' !', '!')
        txt = txt.replace(' ?', '?')
        txt = txt.replace('  ', ' ')

        return txt
    
    def decode(self, txt:str):
        return self._detokenize(txt.split())

    def loadData(self, sample:Union[Tuple[str, str, dict], Any]) -> Tuple[str, str, dict]:
        if isinstance(sample, tuple): return sample
        raise ValueError(sample)

    def createPrompt(self, sample:Union[Tuple[str, str, dict], Any], samples_train:List[Union[str, Tuple[str, str, dict]]]=[]):
        # add prefix:
        prompt  = [self.prefix]

        # add few-shot samples:
        for sample_train in samples_train:
            txt, label, data = self.loadData(sample_train)
            txt = txt.replace('"', '')
            prompt.append(f"\"{txt}\"{self.suffix}{label}")

        # add test sample:
        txt, label, data = self.loadData(sample)
        txt = txt.replace('"', '')
        prompt.append(f"\"{txt}\"{self.suffix}")

        return self.separator.join(prompt), label, data

class PromptLoaderIMDB(PromptLoader):
    def loadData(self, sample:Union[Tuple[str, str, dict], str]) -> Tuple[str, str, dict]:
        if isinstance(sample, tuple): return sample

        # decode json:
        sample = json.loads(sample)[0]

        return (
            self._detokenize(sample["text"].split()),
            'Positive' if sample["classification"] == 'POS' else 'Negative',
            sample["evidences"]
        )

class PromptLoaderFoodIncidents(PromptLoader):
    def loadData(self, sample:Union[Tuple[str, str, dict], Tuple[str, str]]) -> Tuple[str, str, dict]:
        #assert isinstance(sample, tuple), sample
        try: title, spans_string = sample
        except: return sample

        # parse spans string:
        spans = []
        for span in spans_string.split('|'):
            i, j = eval(span)
            spans.append((i, j+1))

        return (
            title,
            title[spans[0][0]:spans[0][1]],
            spans
        )