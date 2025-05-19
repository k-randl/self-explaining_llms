import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from difflib import SequenceMatcher
from transformers import PreTrainedModel, PreTrainedTokenizer
from scipy.stats import pearsonr

from .modelling import no_explain
from typing import List, Dict, Iterable, Tuple, Optional, Union

#=======================================================================#
# Functions:                                                            #
#=======================================================================#

def create_perturbation_curve(results:List[Dict], method:str, direction:str, n_bins=100, average:bool=True):
    # extract x values (fraction of masked tokens):
    xs = np.arange(n_bins + 1, dtype=float) / (n_bins + 1.)

    # extract y values (probability of changed output):
    ys = np.full((len(results), n_bins + 1), np.nan, dtype=float)
    for k, r in enumerate(results):
        points = iter(r['perturbation'][method][direction])
        def get_next():
            point = next(points)
            return point['p'], float(r['prediction']['text'] in point['s'].lower())
        last_x, last_y = 0., 1.
        next_x, next_y = get_next()
        for l, x in enumerate(xs):
            while x > next_x or (next_x - last_x == 0):
                last_x, last_y = next_x, next_y
                next_x, next_y = get_next()
            ys[k,l] = ((next_y - last_y) / (next_x - last_x) * (x - last_x)) + last_y

    if average: return xs, 1. - np.nanmean(ys, axis=0)
    else: return [xs[~np.isnan(ys[i])] for i in range(len(ys))], [1. - ys[i, ~np.isnan(ys[i])] for i in range(len(ys))]

def maskImportance(importance:torch.Tensor, sample_start:int, sample_end:int):
    # copy importance:
    importance = importance.detach().clone()

    # set token importance outside the sample to zero:
    importance[:sample_start] = 0.
    importance[sample_end:]   = 0.

    # return normalized version:
    return importance / importance.sum()

def findSpan(s1:str, s2:str, tokenizer:PreTrainedTokenizer) -> Tuple[int, int]:
    # return on empty string:
    if s1 == '' or s2 == '': return -1, -1

    # find substring:
    start = s1.lower().find(s2.lower())
    end   = start + len(s2)

    # return if not found:
    if start < 0: return -1, -1

    # find token indices:
    offsets = np.array(tokenizer(s1, return_offsets_mapping=True).offset_mapping)
 
    # fix for llama3's bug:
    offsets[:-1,1] = offsets[1:,0]
    offsets[ -1,1] = len(s1)

    indices = np.argwhere((offsets[:,0] < end) & (offsets[:,1] > start))[:,0]

    return indices.min(), indices.max() + 1

#=======================================================================#
# Perturbation:                                                         #
#=======================================================================#

class PerturbationTester:
    def __init__(self, input_ids:torch.Tensor, sample_start:int, sample_end:int, mask_token:int, model:PreTrainedModel, tokenizer:PreTrainedTokenizer, **kwargs):
        bs, self.seq_size = input_ids.shape
        assert bs == 1

        self.sample_size = sample_end - sample_start
        assert self.sample_size > 0

        # set properties:
        self.input_ids    = input_ids
        self.mask_token   = mask_token
        self.sample_start = sample_start
        self.sample_end   = sample_end
        self.model        = model
        self.tokenizer    = tokenizer

        self.model_kwargs = kwargs

        # run complete permutation only once:
        input_ids = input_ids.detach().clone()
        input_ids[:,sample_start:sample_end] = mask_token

        # generate:
        with torch.no_grad():
            with no_explain(model):
                output_ids=model.generate(input_ids=input_ids, **kwargs)[0,self.seq_size:]

        # save:
        self.complete = {'p': 1., 's': self.tokenizer.decode(output_ids)}

    def _perturbe(self, importance:torch.FloatTensor, probability_sampling:bool=True, quantile_sampling:bool=False):
        importance = importance[self.sample_start:self.sample_end].detach().clone()
        importance += 1e-9
        importance /= importance.sum()

        idx_sorted = torch.argsort(importance, descending=True)

        t = .1
        ps, input_ids = [], []
        for i in range(self.sample_size):
            idx = idx_sorted[:i+1].clone()

            if quantile_sampling: p = float(importance[idx].sum().detach().cpu())
            else: p = float(i+1) / float(self.sample_size)
            if p < t: continue
            ps.append(p)

            # sample indices from distribution:
            if probability_sampling:
                idx_max = torch.argwhere(importance == importance[idx][-1]).flatten()
                if len(idx_max) > 1:
                    n_max = (importance[idx] == importance[idx][-1]).sum()
                    idx[-n_max:] = idx_max[torch.randperm(len(idx_max))[:n_max]]

            idx += self.sample_start

            # copy and mask important tokens:
            input_ids.append(self.input_ids.detach().clone())
            input_ids[-1][:,idx] = self.mask_token

            # next threshold:
            t += .1
            if t >= 1: break

        # create attention mask:
        input_ids = torch.concatenate(input_ids, dim=0)
        attention_mask = torch.ones_like(input_ids)
        
        # generate:
        with torch.no_grad():
            with no_explain(self.model):
                output_ids=self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **self.model_kwargs)[:,self.seq_size:]

        # create pad token mask:
        padding_mask = torch.cumsum(output_ids == self.tokenizer.eos_token_id, dim=1) == 0

        return [{'p': p, 's': self.tokenizer.decode(oids[m])} for p, oids, m in zip(ps, output_ids, padding_mask)]

    def test(self, importance:torch.Tensor, double_sided:bool=False, probability_sampling:bool=True, quantile_sampling:bool=False):
        # make sure we get a one-dimensional tensor:
        assert importance.dim() == 1, importance.dim()
        
        # convert importance to probability:
        if double_sided: importance = torch.abs(importance)
        else:            importance = torch.maximum(importance, torch.tensor(1e-9))

        # return perturbations for both directions:
        return {
            'low2high': self._perturbe(1.- importance, probability_sampling=probability_sampling, quantile_sampling=quantile_sampling) + [self.complete,],
            'high2low': self._perturbe(importance, probability_sampling=probability_sampling, quantile_sampling=quantile_sampling) + [self.complete,],
            'random':   self._perturbe(torch.ones_like(importance), probability_sampling=True, quantile_sampling=False) + [self.complete,]
        }

    def testSpans(self, spans:Iterable[Tuple[int,int]]):
        # convert spans to probability:
        importance = torch.zeros(self.seq_size, dtype=float)
        for i,j in spans: importance[i:j] = 1.

        # return perturbations for both directions:
        return {
            'low2high': self._perturbe(1.- importance, probability_sampling=True) + [self.complete,],
            'high2low': self._perturbe(importance, probability_sampling=True) + [self.complete,],
            'random':   self._perturbe(torch.ones_like(importance), probability_sampling=True) + [self.complete,]
        }

#=======================================================================#
# Counterfactuals:                                                      #
#=======================================================================#

def similarityTest(token_ids1:torch.Tensor, token_ids2:torch.Tensor):
    # calculate overlap:
    matcher = SequenceMatcher(None, token_ids1[0].detach().cpu().tolist(), token_ids2[0].detach().cpu().tolist())

    # get replaced tokens in original string:
    replaced = []
    i0 = 0
    for i1, _, n in matcher.get_matching_blocks():
        if i0 != i1: replaced.append((i0,i1))
        i0 = i1 + n

    return matcher.ratio(), replaced

#=======================================================================#
# Correlation:                                                          #
#=======================================================================#

class PearsonCorrelationTester:
    def __init__(self) -> None:
        self.variables = {}

    def add(self, values:Union[List[float], Dict[str,List[float]]], label:str, model:str='') -> None:
        if isinstance(values, dict):
            for m in values:
                self.add(values=values[m], label=label, model=m)

        else:
            if model not in self.variables: self.variables[model] = {}
            self.variables[model][label] = values

    def correlate(self, xs:Union[str,List[float]], ys:Union[str,List[float]], model:Optional[str]=None):
        if isinstance(xs, str): 
            assert model is not None
            xs = self.variables[model][xs]

        if isinstance(ys, str):
            assert model is not None
            ys = self.variables[model][ys]

        n = len(xs)
        assert n == len(ys)

        # compute textwise correlation:
        p = np.empty(n, dtype=float)
        r = np.empty(n, dtype=float)
        for i, (x, y) in enumerate(zip(xs, ys)):
            r[i], p[i] = pearsonr(x, y)

        return r, p

    def boxplot(self, var:Union[str,List[float]], other:Union[Iterable[str],Iterable[List[float]],None]=None, path:Optional[str]=None, legend:bool=True, handles:Optional[List[Tuple[str,Patch]]]=None):
        path = path.replace(' ', '')

        if len(self.variables) > 5:
            model_sizes = list(set([m.split('-')[1] for m in self.variables]))
            model_sizes.sort()
            path, filetype = path.split('.')

            n = len(model_sizes)
            limits = list(range(0, n+1, n // 2))
            for i, j in zip(limits[:-1], limits[1:]):
                sz = model_sizes[i:j]
                plotter = PearsonCorrelationTester()
                for s in sz: plotter.variables.update({
                    m:self.variables[m] for m in self.variables if m.endswith(s)
                })
                
                handles = plotter.boxplot(var=var, other=other, path=f'{path}-{sz[0]}-{sz[-1]}.{filetype}', handles=handles)

            return

        plt.figure(figsize=(8,2))
        if handles is None: labels, handles = [], []
        else:
            labels, handles = tuple(list(item) for item in  zip(*handles))
            for _ in range(len(handles)): plt.gca()._get_lines.get_next_color()

        for model in self.variables:
            # if other is unspecified compare to all:
            if other == None: other = [key for key in self.variables[model] if key != var]

            # calculate correlations:
            rs = []
            ls = []

            for s in other:
                r, p = self.correlate(var, s, model=model)
                rs.append(r[~np.isnan(r)])
                ls.append(f"vs.\n{s}")

            # plot correlations:
            handles.append(Patch(color=plt.violinplot(rs)['bodies'][0].get_fc()))
            labels.append(model)

        plt.hlines([-1., -.5, 0., .5, 1.], .5, len(ls) + .5, colors='lightgrey', zorder=0)
        plt.xlim(left=.5, right=len(ls) + .5)
        plt.xticks(
            ticks    = np.arange(len(ls)) + 1,
            labels   = ls,
        )
        plt.ylim(top=1.1, bottom=-1.1)
        plt.ylabel('Pearson\'s $r$')
        plt.tight_layout()

        # save:
        if path is not None: plt.savefig(path)
        plt.show()

        if legend == True:
            max_cols = 5
            n_labels = len(labels)
            n_rows = ((n_labels//max_cols) + int((n_labels%max_cols) > 0))
            n_cols = ((n_labels//n_rows) + int((n_labels%n_rows) > 0))

            plt.figure(figsize=(9,.3 * n_rows))
            plt.axes().set_axis_off()
            plt.legend(
                handles=handles,
                labels=labels,
                loc='center',
                ncols=n_cols
            )
            plt.savefig('plots/PearsonViolin-Legend.pdf')
            plt.show()

        return list(zip(labels, handles))
        

    def matrixplot(self, vars:Union[List[str],List[List[float]],None]=None, title:str='Pearson r', dir:Optional[str]=None):
        for model in self.variables:
            # if vars is unspecified compare all:
            if vars == None: vars = [key for key in self.variables[model]]

            # calculate correlations:
            rs = np.eye(len(vars), dtype=float)
            ps = np.zeros_like(rs, dtype=float)

            for i in range(len(vars)):
                for j in range(i):
                    r, p = self.correlate(vars[i], vars[j], model=model)
                    
                    ps[i,j] = np.nanmean(p)
                    ps[j,i] = ps[i,j]

                    rs[i,j] = np.nanmean(r)
                    rs[j,i] = rs[i,j]

            # plot correlations:
            plt.imshow(rs, vmin=-1., vmax=1.)
            plt.yticks(
                ticks    = range(len(vars)),
                labels   = vars
            )
            plt.xticks(
                ticks    = range(len(vars)),
                labels   = vars,
                rotation = 90
            )

            # add values:
            for i in range(len(vars)):
                for j in range(i):
                    plt.text(i, j, f'{rs[i,j]:.2f}\n({ps[i,j]:.2f})', ha='center', va='center')
                    plt.text(j, i, f'{rs[i,j]:.2f}\n({ps[i,j]:.2f})', ha='center', va='center')

            plt.title(title)
            plt.tight_layout()

            # save:
            os.makedirs(f'{dir}/{model}', exist_ok=True)
            if dir is not None: plt.savefig(f'{dir}/{model}/{title}.pdf')

            plt.show()