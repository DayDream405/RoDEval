# -*- coding: utf-8 -*-
# @Author: Zly
# experiment executor
"""
This module contains classes related to experiment executors, used to configure and execute various experiments with large language models.
Classes:
    ExperimentExecutor: Abstract base class defining the interface for experiment executors.
    LlmExperimentExecutor: Concrete implementation class that executes experiments based on LLMs.
    LlmExecutorConfig: Configuration class for setting parameters such as large language model, dataset, prompt generator, and number of epochs.
Functions:
    save_result: Saves experimental results to a specified path.

"""
import os
from collections.abc import Container
from typing import List
from abc import ABC, abstractmethod

from entity import results
from entity.experimental_data import ExperimentalData, MCQExperimentalData
from entity.large_language_model import LargeLanguageModel
from entity.results import ExperimentalResult
from experiment.experimenter import ExperimenterInterface, LlmExperimenter, save_record
from utils.dataset_process import DataSetInterface, UnifiedDataSet
from utils.evaluator import EvaluatorInterface
from utils.formatter import *
from utils.prompt_generator import PromptGenerator, PromptFragmentInterface
from stream.file_stream import *

default_dataset_name = ['semeval2007', 'semeval2013', 'semeval2015', 'senseval2', 'senseval3']
default_dataset_path = [['datasets/en/semeval2007/semeval2007.data.xml', 'datasets/en/semeval2007/semeval2007.gold.key.txt'],
                        ['datasets/en/semeval2013/semeval2013.data.xml', 'datasets/en/semeval2013/semeval2013.gold.key.txt'],
                        ['datasets/en/semeval2015/semeval2015.data.xml', 'datasets/en/semeval2015/semeval2015.gold.key.txt'],
                        ['datasets/en/senseval2/senseval2.data.xml', 'datasets/en/senseval2/senseval2.gold.key.txt'],
                        ['datasets/en/senseval3/senseval3.data.xml', 'datasets/en/senseval3/senseval3.gold.key.txt'],
                        ]

class ExperimentExecutor(ABC):
    """ Abstract base class for experiment executors.

    Methods:
        add_experimenter(self, experimenter: ExperimenterInterface): Add an experimenter to the executor.
        execute(self): Execute the experiments.
    """

    @abstractmethod
    def add_experimenter(self, experimenter: ExperimenterInterface):
        pass

    @abstractmethod
    def execute(self):
        pass
    pass

def save_result(experimenter: LlmExperimenter, dataset_name, result: EvaluationResults):
    """ Save experimental results to a specified path.

    Args:
        experimenter (LlmExperimenter): The experimenter used in the experiment.
        dataset_name (str): The name of the dataset used in the experiment.
        result (EvaluationResults): The results of the experiment to be saved.
    """
    if not os.path.exists(r'result/{}/{}'.format(experimenter.llm_name, experimenter.experiment_name)):
        os.makedirs(r'result/{}/{}'.format(experimenter.llm_name, experimenter.experiment_name))

    path = ResultFilePath(experimenter.llm_name, experimenter.experiment_name, dataset_name)
    total_scores = EvaluationResults()
    for k, v in result.results.items():
        if isinstance(v, Container):
            p = ResultFilePath(experimenter.llm_name, experimenter.experiment_name, dataset_name, k)
            container_result = EvaluationResults()
            container_result.add_result(k, v)
            save_results2json(p, container_result)
        else:
            total_scores.add_result(k, v)
    save_results2json(path, total_scores)
    pass


class LlmExperimentExecutor(ExperimentExecutor):
    """ LLM Experiment Executor

    Attributes:
        llm (LargeLanguageModel): The large language model used in the experiments.
        dataset (List[DataSetInterface]): The datasets used in the experiments.
        epoch (int): The number of epochs for the experiments.
        prompt_generator (PromptGenerator): The prompt generator used to create prompts for the LLM.
        prompt_formatter (FormatterController): The formatter controller used to format prompts.
        _experimenter_list (List[LlmExperimenter]): A list of experimenters added to the executor.
    Methods:
        add_experimenter(self, experimenter: LlmExperimenter, evaluator: EvaluatorInterface): Add an experimenter and its evaluator to the executor.
        execute(self, is_save_result=False, is_save_record=False): Execute the experiments with the added experimenters.
    """

    def __init__(self, llm: LargeLanguageModel, dataset: List[DataSetInterface], epoch: int,
                 prompt_generator: PromptGenerator, 
                 prompt_formatter: FormatterController=None):

        self.llm = llm
        self.dataset = dataset
        self.epoch = epoch
        self.prompt_generator = prompt_generator
        self.prompt_formatter = prompt_formatter
        self._experimenter_list: List[LlmExperimenter] = []


    def execute(self, is_save_result=False, is_save_record=False):
        for experimenter in self._experimenter_list:
            for data in self.dataset:
                record, result = experimenter.run_experiment(data, self.epoch, is_save_record)
                if is_save_record:
                    save_record(experimenter, data.name, record, None)
                if is_save_result:
                    save_result(experimenter, data.name, result)


        pass

    def add_experimenter(self, experimenter: LlmExperimenter, evaluator: EvaluatorInterface=None):
        if evaluator is not None:
            experimenter.general_settings(self.llm, self.prompt_generator, evaluator)
        self._experimenter_list.append(experimenter)
        pass


class LlmExecutorConfig(object):
    """ LLM Executor Configuration
    This class is used to configure the parameters for the LLM experiment executor, including the large language model,
    dataset, prompt generator, and number of epochs.
    
    Methods:
        set_llm(self, name: str, interact_function, **kwargs): Set the large language model.
        set_dataset_name(self, name: List[str]): Set the names of the datasets.
        set_data_path_list(self, path_list: List[List[str]]): Set the paths of the datasets.
        choose_default_dataset(self, start: int, end: int): Choose a default dataset from a predefined list.
        set_prompt_generator(self, prompt_fragment_list: List[PromptFragmentInterface]): Set the prompt generator using a list of prompt fragments.
        set_prompt_formatter(self, is_title_case: bool=False, is_upper_case: bool=False,
                             is_remove_punctuation: bool=False, is_lower_case: bool=False): Set the prompt formatter with various formatting options.
        set_epoch(self, epoch: int): Set the number of epochs for the experiments.
        get_executor(self) -> LlmExperimentExecutor | None: Get the configured LLM experiment executor.

    """
    def __init__(self):
        self.__llm = None
        self.__dataset_name: List[str] = default_dataset_name
        self.__data_path_list: List[List[str]] = default_dataset_path
        self.__data_list: List[DataSetInterface] = []
        self.__prompt_generator = None
        self.__prompt_formatter = None
        self.__epoch: int = 1
        self._experimenter_list: List[LlmExperimenter] = []

    def set_llm(self, name: str, interact_function, **kwargs) -> None:
        self.__llm = LargeLanguageModel(name, interact_function, **kwargs)

    def set_dataset_name(self, name: List[str]) -> None:
        self.__dataset_name = name

    def set_data_path_list(self, path_list: List[List[str]]) -> None:
        self.__data_path_list = path_list
    def set_custom_dataset(self, dataset: DataSetInterface=None, data_list: List[DataSetInterface]=None) -> None:
        if dataset is not None:
            self.__data_list = [dataset]
        elif data_list is not None:
            self.__data_list = data_list

    def choose_default_dataset(self, start: int, end: int):
        """ Choose a default dataset from a predefined list. The predefined datasets are:
        ['semeval2007', 'semeval2013', 'semeval2015', 'senseval2', 'senseval3', 'subAll']

        Args:
            0 <= start < end <= 6, [start, end) is the range of indices to choose from the predefined list.
        """
        assert 0 <= start < end <= 6, 'Invalid dataset index range: start={}, end={}'.format(start, end)
        self.__dataset_name = default_dataset_name[start:end]
        self.__data_path_list = default_dataset_path[start:end]

    def set_prompt_generator(self, prompt_fragment_list: List[PromptFragmentInterface]) -> None:
        generator = PromptGenerator()
        for item in prompt_fragment_list:
            generator.add_prompt_fragment(item)
        self.__prompt_generator = generator

    def set_prompt_formatter(self, is_title_case: bool=False, is_upper_case: bool=False,
                             is_remove_punctuation: bool=False, is_lower_case: bool=False) -> None:
        """ Set the prompt formatter with various formatting options.

        Args:
            is_title_case (bool, optional): Defaults to False.
            is_upper_case (bool, optional): Defaults to False.
            is_remove_punctuation (bool, optional): Defaults to False.
            is_lower_case (bool, optional): Defaults to False.
        """
        formatter_controller = FormatterController()
        if is_title_case:
            formatter_controller.add_formatter(TitleCaseFormatter())
        if is_upper_case:
            formatter_controller.add_formatter(UpperCaseFormatter())
        if is_remove_punctuation:
            formatter_controller.add_formatter(RemovePunctuationFormatter())
        if is_lower_case:
            formatter_controller.add_formatter(LowerCaseFormatter())
        self.__prompt_formatter = formatter_controller

    def set_epoch(self, epoch: int) -> None:
        assert epoch >= 1, 'The number of experimental rounds must be greater than 1, setted epoch:{}'.format(epoch)
        self.__epoch = epoch
    
    def add_experimenter(self, experimenter: LlmExperimenter, evaluator: EvaluatorInterface):
        experimenter.general_settings(self.llm, self.prompt_generator, evaluator)
        self._experimenter_list.append(experimenter)
        pass

    def get_executor(self) -> LlmExperimentExecutor | None:
        if len(self.__data_list) == 0:
            if len(self.__dataset_name) != 0:
                assert (len(self.__dataset_name) == len(self.__data_path_list)), 'The number of dataset name does not match the paths. name:{},path:{}' \
                    .format(len(self.__dataset_name), len(self.__data_path_list))
            for i in range(len(self.__dataset_name)):
                data_name = self.__dataset_name[i]
                data_path = self.__data_path_list[i]
                data = UnifiedDataSet(data_name, data_path[0], data_path[1])
                self.__data_list.append(data)

        executor = LlmExperimentExecutor(self.__llm, self.__data_list, self.__epoch,
                                         self.__prompt_generator, self.__prompt_formatter)
        if len(self._experimenter_list) != 0:
            for e in self._experimenter_list:
                executor.add_experimenter(e)
        return executor


def main():
    # configure and run an example experiment
    executor_fig = LlmExecutorConfig()
    # set llm
    executor_fig.set_llm('gpt-4o', interact_function=None)
    # set epoch
    executor_fig.set_epoch(5)
    # choose dataset
    executor_fig.choose_default_dataset(0, 1)
    # set prompt generator
    executor_fig.set_prompt_generator([])
    executor = executor_fig.get_executor()
    # set experimenter and evaluator
    experimenter = LlmExperimenter()
    evaluator = None
    executor.add_experimenter(experimenter, evaluator)
    # run
    executor.execute(is_save_record=True, is_save_result=True)
    pass

if __name__ == '__main__':
    main()
