# -*- coding: utf-8 -*-
# @Author: Zly
#results
"""
Evaluation results entity class
"""

from typing import Dict, List
from collections.abc import Container

from entity.experimental_data import ExperimentalData, Polysemous
from utils.tools import merge_dicts, log2rate


class EvaluationResults(object):
    """
    EvaluationResults class for storing various evaluated results.

    Attributes:
        _result (Dict): Dictionary for storing experimental results
    """

    def __init__(self):
        self._result: Dict = {}

    @property
    def results(self) -> Dict:
        return self._result

    def add_result(self, result_key: str, result: List | Dict) -> object | None:
        """add or update a result in the results dictionary. If the result_key already exists, it will overwrite the existing result and return the old result. If the result_key does not exist, it will add the new result and return None.

        Args:
            result_key(str): The name of the result also serves as the key to get this result.
            result(List|Dict): The result to be stored, can be a List or Dict.

        Returns:
            object: If the result_key already exists, return the old result. If it does not exist, return None.

        """

        if result_key in self._result:
            old_result = self._result[result_key]
            self._result[result_key] = result
            return old_result
        else:
            self._result[result_key] = result
            return None

    def get_result(self, result_key: str) -> object | None:
        """get a result from the results dictionary by its key. If the key does not exist, return None.

        Args:
            result_key: str: The name of the result to be retrieved.

        Returns:
            object: The result corresponding to the result_key. If the key does not exist, return None.

        """

        return self._result.get(result_key, None)

    def __str__(self):
        """Print experiment results in Dictionary

        Returns:
            str: result_name1: result1
                 result_name2: result2...
        """
        s = ''
        for k, v in self._result.items():
            if not isinstance(v, Container):
                s += '{}: {}\n'.format(k, v)
            else:
                s += '{}: {}(Omitted)\n'.format(k, type(v))
        return s

    pass


class ExperimentalResult(ExperimentalData):
    """ ExperimentalResult class for storing experimental results.

        Attributes:
            _pre(List): Model prediction results
            _real(List): Correct answers (optional, for tasks without correct answers)
        
        Inherits from ExperimentalData class to include experimental data attributes.


    """
    def __init__(self, experimental_data: ExperimentalData,
                 pre: str, real: str = None):
        
        super().__init__(experimental_data.polysemous, experimental_data.context, experimental_data.target,
                         experimental_data.correct_definition_in_contexts, experimental_data.correct_definition_keys)
        self._pre: str = pre
        self._real: str|None = real

    @property
    def pre(self) -> str:
        return self._pre

    @property
    def real(self) -> str|None:
        return self._real
    pass


class MemoryProbeResults(ExperimentalResult):
    """ MemoryProbeResults class for storing memory probe experimental results.

        Attributes:
            _logprob: List: List of token log probabilities from the model's output.
        Inherits from ExperimentalResult class to include experimental result attributes.
    """
    def __init__(self, experimental_data: ExperimentalData, logprob,
                 pre: str, real: str = None):
        super().__init__(experimental_data, pre, real)
        self.__logprob = logprob
    
    @property
    def token_logits(self):
        """ Get token logits from the model's output.

        Returns:
            List: [[(token1.1, p1.1),(token1.2, p1.2)...][(token2.1, p2.1...)]]
        """
        logits_list = []
        for token in self.__logprob:
            logits = []
            logs = token.top_logprobs
            for log in logs:
                p = log2rate(log.logprob)
                logits.append((log.token, p))
            logits_list.append(logits)
        return logits_list

def merge_evaluationresults(result_list: List[EvaluationResults]) ->EvaluationResults:
    """Merge multiple EvaluationResults into one EvaluationResults.

    Args:
        result_list (List[EvaluationResults]): List of EvaluationResults to be merged.

    Returns:
        EvaluationResults: Merged EvaluationResults.
    """
    merged_evaluationresult = EvaluationResults()
    data_dicts = [data for data in result_list]
    merged_dict = merge_dicts(data_dicts)
    for k, v in merged_dict.items():
        merged_evaluationresult.add_result(k, v)
    
    return merged_evaluationresult

