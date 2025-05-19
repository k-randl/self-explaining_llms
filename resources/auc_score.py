import warnings
import numpy as np
import pandas as pd

from .testing import create_perturbation_curve
from typing import List

class AUCScorer:
    def __init__(self, results:List) -> None:
        self._results  = results

    def score(self, model:str, method:str, direction:str, n_bins:int=100, average:bool=True):
        x, y = create_perturbation_curve(self._results[model], method, direction, n_bins=n_bins, average=average)

        if average: return np.trapz(y, x)
        else: return np.array([np.trapz(y[i], x[i]) for i in range(len(x))])

    def score_table(self, n_bins:int=100, average:bool=True):
        result = {}

        for model in self._results:
            result[model] = {}

            for method in self._results[model][0]['perturbation']:
                high2low = self.score(model, method, 'high2low', n_bins=n_bins, average=average)
                low2high = self.score(model, method, 'low2high', n_bins=n_bins, average=average)
                result[model][method] = high2low - low2high

        result = pd.DataFrame(data=result)

        return result
