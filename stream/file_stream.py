# -*- coding: utf-8 -*-
# @Author: Zly
# log
"""
Input/output stream (file stream)
Handles reading from and writing to files.
Provides functionality to save and load experiment results in CSV and JSON formats.
"""
import csv
import json
from typing import List, Dict

from entity.results import EvaluationResults

class ResultFilePath(object):
    """File path management for experiment results.
    Constructs file paths based on model name, experiment name, dataset name, and optional additional arguments.
    Example:
        path = ResultFilePath('gpt-4o', 'mcq_experiment', 'dataset1', 'extra1', 'extra2')
        print(path)  # Outputs: result/gpt-4o/mcq_experiment/dataset1_extra1_extra2
        csv_path = path.add_extension('csv')  # Outputs: result/gpt-4o/mcq_experiment/dataset1_extra1_extra2.csv
    """

    def __init__(self,model_name: str, experiment_name: str, dataset_name: str, *args):
        self.__model_name = model_name
        self.__experiment_name = experiment_name
        self.__dataset_name = dataset_name
        self.__args = args

    @property
    def model_name(self):
        return self.__model_name
    @property
    def experiment_name(self):
        return self.__experiment_name
    @property
    def dataset_name(self):
        return self.__dataset_name

    def get_path(self):
        return self.__str__()

    def __str__(self):
        dt = ''
        if len(self.__args) != 0:
            for s in self.__args:
                dt += '_' + s

        return r'result/{}/{}/{}{}'.format(self.model_name, self.experiment_name, self.dataset_name, dt)

    def add_extension(self, extension: str) -> str:
        return r'{}.{}'.format(self.__str__(), extension)
    pass


def save_results2csv(path: ResultFilePath, results_list: List[Dict], mode='w') -> None:
    """ Save experiment results to a CSV file.

    Args:
        path (ResultFilePath): The ResultFilePath object.
        results_list (List[Dict]): List of dictionaries containing the results.
        mode (str, optional): save mode('w', 'a'). Defaults to 'w'.
    """
    assert (mode == 'w' or 'a'), 'Only allow write and append modes (w, a)'

    header = True if mode == 'w' else None
    path_str = path.add_extension('csv')
    with open(path_str, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=results_list[0].keys(), dialect='excel')
        if header:
            writer.writeheader()
        for r in results_list:
            writer.writerow(r)
    print('结果已存至{}'.format(path_str))
    pass

def read_results_from_csv(path: ResultFilePath) -> List[Dict]:
    """ Read experiment results from a CSV file.

    Args:
        path (ResultFilePath): The ResultFilePath object.

    Returns:
        List[Dict]: List of dictionaries containing the results.
    """
    path_str = path.add_extension('csv')
    with open(path_str, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, dialect='excel')
        results = []
        for row in reader:
            d = {}
            for k, v in row.items():
                d[k] = v
            results.append(d)
        return results

def save_results2json(path: ResultFilePath, results: EvaluationResults, mode='w') -> None:
    """ Save experiment results to a JSON file.

    Args:
        path (ResultFilePath): The ResultFilePath object.
        results (EvaluationResults): The EvaluationResults object containing the results.
        mode (str, optional): save mode('w', 'a'). Defaults to 'w'.
    """
    assert (mode == 'w' or 'a'), 'Only allow write and append modes (w, a)'

    path_str = path.add_extension('json')
    result_dict = results.results
    with open(path_str, mode=mode, newline='', encoding='utf-8') as file:
        json.dump(result_dict, file, indent=4, ensure_ascii=False)
    print('结果已存至{}'.format(path_str))
    pass

def read_result_from_json(path: ResultFilePath) -> EvaluationResults:
    """ Read experiment results from a JSON file.

    Args:
        path (ResultFilePath): The ResultFilePath object.

    Returns:
        EvaluationResults: The EvaluationResults object containing the results.
    """
    path_str = path.add_extension('json')
    
    with open(path_str, mode='r', newline='', encoding='utf-8') as file:
        data = json.load(file)
        r = EvaluationResults()
        for k, v in data.items():
            r.add_result(k, v)
        return r