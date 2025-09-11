# -*- coding: utf-8 -*-
# @Author: Zly
# utils for analysis
""" This module provides various analysis functions for analyzing experimental results,
    including:
    1. Dres for RodEval
    2. WSD Scope for RodEval
    3. Dros for RodEval
    4. Word-level accuracy merging across multiple experiments
"""
from typing import List, Dict
import os
import math
import numpy as np
from itertools import combinations
import copy
from collections import defaultdict
import hashlib
import random
import tqdm
from collections import Counter

from utils.wordnet_tools import wn, WordNetSynsets
from stream.file_stream import *
from entity.results import EvaluationResults, ExperimentalResult, ExperimentalData, Polysemous
from entity.large_language_model import LargeLanguageModel
def Dres(pre_results: List[Dict], real_results: List[Dict], p_a: float, p_b: float):
    """Dres for RodEval

    Args:
        pre_results (List[Dict]): Prediction results of model self-check, e.g., [{'pre': 'yes'}, ...]
        real_results (List[Dict]): Real results of model self-check, e.g., [{'real': '1'}, ...]
        p_a (float): Probability of answering correctly P(A)
        p_b (float): Probability of answering "yes" P(B)
    """
    p_ba = 0 
    p_b_cond_a = 0 # P(B|A)
    data_quantity = len(pre_results)
    for i, real in enumerate(real_results):
        pre = pre_results[i]
        if real['real'] in real['pre'] and 'yes' in pre['pre'].lower(): # AcapB
            p_ba += 1
    p_ba /= data_quantity # P(AcapB)
    p_b_cond_a = p_ba / p_a # P(B|A)
    p_a_cond_b = (p_b_cond_a * p_a) / p_b # P(A|B)
    return p_a_cond_b

def scope(llm_name: str, dataset_name: str, interval_size: int=25):
    """ Calculate WSD scope for RodEval

    Args:
        llm_name (str): llm_name
        dataset_name (str): dataset_name
        interval_size (int, optional): Accuracy interval size. Defaults to 25.
    """
    experiment_list = ['classification-normal', 'classification-op1', 'classification-op2', 'classification-op3', 'classification-op4']
    sense_acc_list: List[Dict] = []
    for experiment_name in experiment_list:
        path = ResultFilePath(llm_name, experiment_name, dataset_name, 'sense_accuracy')
        data = read_result_from_json(path).get_result('sense_accuracy')
        sense_acc_list.append(data)
    merge_sense: Dict= word_accuracy(sense_acc_list)
    
    results = EvaluationResults()
    word_types_number = len(merge_sense)
    for threshold in range(interval_size, 100+interval_size, interval_size):
        count = 0
        for k, v in merge_sense.items():
            if v * 100 >= threshold:
                count += 1
        results.add_result('scope{}'.format(threshold), count / word_types_number)
    folder_path = r'result/{}/wsd_scopes'.format(llm_name) # ensure the directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    path = ResultFilePath(llm_name, 'wsd_scopes', dataset_name, 'scope')
    save_results2json(path, results)
    scopes = EvaluationResults()
    scopes.add_result('scopes', merge_sense)
    path = ResultFilePath(llm_name, 'wsd_scopes', dataset_name, 'word_scopes')
    save_results2json(path, scopes)

def dros(data_list: List):
    """ Calculate Dros for RodEval
    Args:
        data_list (List): List of accuracy values for different words
    """
    n = len(data_list)
    avg = sum(data_list) / n
    step = 0
    for d in data_list:
        step += math.pow(d - avg, 2)
    step /= n
    step = math.sqrt(step)
    return 1 - step

def word_accuracy(sense_acc_list: List[Dict]) -> Dict:
    """ Merge word-level accuracy across multiple experiments
    Args:
        sense_acc_list (List[Dict]): List of sense accuracy dictionaries from different experiments
    Returns:
        Dict: Merged word-level accuracy dictionary
    """
    merge_sense: Dict={}
    for sa in sense_acc_list:
        for k, v in sa.items():
            lemma = wn.lemma_from_key(k).name()
            if lemma in merge_sense:
                merge_sense[lemma].append(v)
            else:
                merge_sense[lemma] = [v]
    for k, v in merge_sense.items():
        merge_sense[k] = sum(v) / len(v)
    return merge_sense
