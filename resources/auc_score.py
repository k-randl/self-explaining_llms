import warnings
import numpy as np
import pandas as pd

from typing import List

class AUCScorer:
    def __init__(self, results:List) -> None:
        self._results  = results

    def score(self, model:str, method:str, direction:str, n_bins:int=6):
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
            x = [0,] + [np.mean(x[i==j]) for j in range(1,len(bins))]
            y = [0,] + [np.mean(y[i==j]) for j in range(1,len(bins))]

        # integrate:
        return np.trapz(y, x)

    def score_table(self, n_bins:int=6):
        result = {}

        for model in self._results:
            result[model] = {}

            for method in self._results[model][0]['perturbation']:
                high2low = self.score(model, method, 'high2low', n_bins)
                low2high = self.score(model, method, 'low2high', n_bins)
                result[model][method] = high2low - low2high

        result = pd.DataFrame(data=result)

        return result
