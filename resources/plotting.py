import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

from typing import Iterable, List, Dict, Union, Optional

class Plotter:
    def __init__(self, results:List, i_sample:Optional[int]=None, save_dir:Optional[str]=None) -> None:
        self._results  = results
        self._save_dir = save_dir
        self._i_sample = i_sample
        self._n_sample = min([len(results[key]) for key in results])

    def get_sample(self) -> int:
        if self._i_sample is None: return random.randint(0,self._n_sample) - 1
        else:                      return self._i_sample

    def token2latex(self, t:str):
        t = t.replace('"', '')
        t = t.replace('_', '')
        t = t.replace('Ċ', '')
        t = t.replace('Ġ', '')
        return f'$\\tt\u007b{t}\u007d$'

    def plot_importance(self, values:Iterable[Union[np.ndarray, Dict[str, np.ndarray]]], labels:Iterable[str], title:str):
        i = self.get_sample()

        for model in self._results:
            r = self._results[model][i]

            start = r['sample']['start']
            end   = r['sample']['end']

            vs = []
            ls = []

            for v, l in zip(values, labels):
                v = v[model][i]

                if isinstance(v, dict):
                    vs.extend([v[key] for key in v])
                    ls.extend([l+self.token2latex(key) for key in v])

                else:
                    vs.append(v)
                    ls.append(l)

            vs = np.array(vs)
            plt.imshow(vs)
            plt.xticks(
                ticks    = range(end-start),
                labels   = [self.token2latex(l) for l in r['tokens'][start:end]],
                rotation = 90
            )
            plt.yticks(
                ticks    = range(len(ls)),
                labels   = ls
            )
            plt.title(title + ' - ' + model)
            plt.tight_layout()

            dir = f'{self._save_dir}/{model}'
            os.makedirs(dir, exist_ok=True)
            if self._save_dir is not None: plt.savefig(f'{dir}/{title}.pdf')
            plt.show()

    def plot_perturbation(self, title:str, n_bins:int=6, legend:bool=True, top:float=1., bottom:float=0.):
        fig, axs = plt.subplots(nrows=2, ncols=len(self._results), figsize=(7,4.75))

        def plot_curve(ax, method, direction, **kwargs):
            # extract x values (fraction of masked tokens):
            x = np.array([[point['p'] for point in r['perturbation'][method][direction]]
                for r in self._results[model]]).flatten()

            # extract y values (probability of changed output):
            y = 1. - np.array([[r['prediction']['text'] in point['s'].lower() for point in r['perturbation'][method][direction]]
                for r in self._results[model]]).flatten().astype(float)
            
            # bin along x-axis:
            bins = np.linspace(0., 1., n_bins) + .1
            i = np.digitize(x, bins)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                x = [0,] + [np.mean(x[i==j])*100. for j in range(1,len(bins))]
                y = [0,] + [np.mean(y[i==j])*100. for j in range(1,len(bins))]

            # plot curves:
            ax.plot(x, y, **kwargs)

        for i, direction in enumerate(['high2low', 'low2high']):
            for j, model in enumerate(self._results):
                for label in self._results[model][0]['perturbation']:
                    if label != 'human': plot_curve(axs[i,j], label, direction, label=label)

                plot_curve(axs[i,j], 'human', direction, label='human', linestyle='dashed')
                plot_curve(axs[i,j], 'human', 'random',  label='random', linestyle='dashed')

                ax2 = axs[i,j].twinx()

                axs[i,j].grid(visible=True)
                axs[i,j].set_ylim(top=100*top+1, bottom=100*bottom-1)
                ax2.set_ylim(top=100*top+1, bottom=100*bottom-1)
                axs[i,j].set_xlim(left=-1, right=101)
                
                axs[0,j].arrow(50, 50 * (top-bottom) + 100 * bottom, -30, 30 * (top-bottom), width=5, head_length=10, ec='white', color='lightblue')
                axs[0,j].text(35, 60 * (top-bottom) + 100 * bottom, 'better', ha='center', va='bottom', rotation=-45, color='lightblue', zorder=0)

                axs[1,j].arrow(50, 50 * (top-bottom) + 100 * bottom, 30, -30 * (top-bottom), width=5, head_length=10, ec='white', color='lightblue')
                axs[1,j].text(65, 30 * (top-bottom) + 100 * bottom, 'better', ha='center', va='bottom', rotation=-45, color='lightblue', zorder=0)

                axs[0,j].set_title(model, fontweight ="bold")
                axs[0,j].set_xticklabels([])
                axs[1,j].set_xlabel('Masked Tokens [%]')
                axs[i,j].set_box_aspect(1.)
                
                if j == 0:
                    axs[i,j].set_ylabel(f'{direction.replace("2", " to ")} imp.', fontweight ="bold")
                    ax2.set_yticklabels([])

                elif j + 1 == len(self._results):
                    axs[i,j].set_yticklabels([])
                    ax2.set_ylabel('Changed Outputs [%]')

                else:
                    axs[i,j].set_yticklabels([])
                    ax2.set_yticklabels([])
                

        fig.suptitle(title)
        plt.tight_layout()

        if self._save_dir is not None: plt.savefig(f'{self._save_dir}/{title}.pdf')
        plt.show()

        if self._save_dir is not None and legend:
            handles, labels = axs[0,0].get_legend_handles_labels()

            plt.figure(figsize=(9,.3))
            plt.axes().set_axis_off()
            plt.legend(
                handles=handles,
                labels=labels,
                loc='center',
                ncols=len(handles)
            )
            plt.savefig(f'{self._save_dir}/Faithfullness - Legend.pdf')
            plt.show()

    def print_chat(self, model:str):
        i = self.get_sample()
        r = self._results[model][i]

        print('\\begin{tabular}{p{.05\\linewidth}p{.1\\linewidth}p{.8\\linewidth}}')
        for role, text in r['chat']:
            if role == 'user': print('\n\\hline\n')
            print('&')
            print(f'\\textbf\u007b{role}\u007d: &')
            print(f'\\texttt\u007b{text}\u007d \\\\\n')
        print('\n\\hline\n')
        print('\\end{tabular}')