# -*- coding: utf-8 -*-
# @Author: Zly
# experimenter
""" 
This module contains experimenter classes for conducting various types of LLM WSD experiments.
The experimenter classes include:
- ExperimenterInterface: An interface defining the basic methods for experimenters.
- LlmExperimenter: A base class for large language model experimenters, inheriting from ExperimenterInterface, containing common experiment settings and execution methods.
- McqExperimenter: A multiple-choice question experimenter, inheriting from LlmExperimenter, LLMused for conducting multiple-choice experiments.
- GenerationExperiment: A generation experimenter, inheriting from LlmExperimenter, used for conducting generation experiments.
- ContaminationTraceExperimenter: A contamination trace experimenter, inheriting from LlmExperimenter, used for conducting contamination trace experiments.
- SelfCheckExperimenter: A self-check experimenter, inheriting from LlmExperimenter, used for conducting self-check experiments.
Each experimenter class implements the run_experiment method for executing specific experimental logic.
"""

from datetime import datetime
from abc import ABC, abstractmethod
from typing import List
import time
import tqdm
import random
import os

from entity.large_language_model import LargeLanguageModel
from entity.experimental_data import ExperimentalData, MCQExperimentalData
from entity.results import ExperimentalResult, EvaluationResults, merge_evaluationresults
from utils.dataset_process import DatasetInterface
from utils.formatter import FormatterController
from utils.prompt_generator import PromptGenerator, LeadingWordAndLeadedContent
from utils.evaluator import EvaluatorInterface, MCQEvaluator, GenerationScoreEvaluator, TraditionalGenerationCriteriaEvaluator
from utils.word_definition_options_generator import OptionGeneratorInterface
from stream.log import load_logger
from utils.en_prompts import GuidedInstructionRequirementEn, GenernalInstructionRequirementEn, SelfCheckInstructionRequirementEn
from stream.file_stream import *

class ExperimenterInterface(ABC):

    @abstractmethod
    def run_experiment(self, dataset: DatasetInterface, epoch: int = 1):
        pass

    pass

class LlmExperimenter(ExperimenterInterface):
    """ Conduct a LLM WSD experiment


    Attributes:
        experiment_name(str): experiment name
        _llm(LargeLanguageModel): LLM used in the experiment
        _prompt_generator(PromptGenerator): Prompt generator
        _prompt_formatter(FormatterController): Prompt formatter
        _evaluator(EvaluationInterface): Evaluator

    Methods:
        run_experiment(self, dataset: DataSetInterface, epoch: int=1): Run the experiment
        general_settings(self, llm: LargeLanguageModel,
                         prompt_generator: PromptGenerator, evaluator: EvaluatorInterface,
                         prompt_formatter: FormatterController=None): Set general settings for the experiment
        experiment_name(self) -> str: Get the experiment name
        llm_name(self) -> str: Get the LLM name
        evaluator(self) -> EvaluatorInterface | None: Get the evaluator
        evaluator(self, evaluator: EvaluatorInterface) -> None: Set the evaluator

    """

    def __init__(self, experiment_name: str):
        self._experiment_name: str = experiment_name
        self._llm = None
        self._prompt_generator = None
        self._prompt_formatter = None
        self._evaluator = None

        self._context = LeadingWordAndLeadedContent('Context')
        self._target_word = LeadingWordAndLeadedContent('TargetWord')
        self._options = LeadingWordAndLeadedContent('Options')
        self._output = LeadingWordAndLeadedContent('Output')

    def run_experiment(self, dataset: DatasetInterface, epoch: int=1, is_save_record=False):
        pass
    def general_settings(self, llm: LargeLanguageModel,
                         prompt_generator: PromptGenerator, evaluator: EvaluatorInterface,
                         prompt_formatter: FormatterController=None):
        self._llm = llm
        self._prompt_generator = prompt_generator
        self._evaluator = evaluator
        self._prompt_formatter = prompt_formatter
        pass

    @property
    def experiment_name(self) -> str:
        return self._experiment_name
    @property
    def llm_name(self) -> str:
        return self._llm.name
    @property
    def evaluator(self) -> EvaluatorInterface | None:
        return self._evaluator
    @evaluator.setter
    def evaluator(self, evaluator: EvaluatorInterface) -> None:
        self._evaluator = evaluator
        pass

def save_record(experimenter: LlmExperimenter, dataset_name: str, record: List[ExperimentalResult], epoch=None):
    """ Save experimental records to a CSV file.

    Args:
        experimenter (LlmExperimenter): The experimenter conducting the experiment.
        dataset_name (str): name of the dataset used in the experiment.
        record (List[ExperimentalResult]): List of current epoch's experimental results to be saved.
        epoch (_type_, optional): current epoch. Defaults to None.
    """
    if not os.path.exists(r'result/{}/{}'.format(experimenter.llm_name, experimenter.experiment_name)):
        os.makedirs(r'result/{}/{}'.format(experimenter.llm_name, experimenter.experiment_name))
    if epoch is not None:
        path = ResultFilePath(experimenter.llm_name, experimenter.experiment_name, dataset_name, epoch)
    else:
        path = ResultFilePath(experimenter.llm_name, experimenter.experiment_name, dataset_name)
    record_dicts = []
    for r in record:
        record_dicts.append({'pre': r.pre, 'real': r.real})
    save_results2csv(path, record_dicts)
    pass

class McqExperimenter(LlmExperimenter):
    """ Multiple Choice Question Experimenter

    Attributes:
        experiment_name(str): experiment name
        _options_generator(OptionGeneratorInterface): Options generator for multiple-choice questions
    Methods:
        run_experiment(self, dataset: DataSetInterface, epoch: int=1): Run the multiple-choice experiment
    """

    def __init__(self, experiment_name: str, options_generator: OptionGeneratorInterface):
        super().__init__(experiment_name)
        self._options_generator: OptionGeneratorInterface = options_generator

    def run_experiment(self, dataset: DatasetInterface, epoch: int=1, is_save_record=False) -> tuple[List[ExperimentalResult], EvaluationResults]|None:
        
        self._prompt_generator.add_prompt_fragment(self._context)
        self._prompt_generator.add_prompt_fragment(self._target_word)
        self._prompt_generator.add_prompt_fragment(self._options)
        self._prompt_generator.add_prompt_fragment(self._output)

        experimental_results: List[ExperimentalResult] = []
        mcq_data_list: List[MCQExperimentalData] = []
        for data in dataset.data_list:
            option_list = self._options_generator.generate_options(data)
            mcq_data = MCQExperimentalData(data, option_list)
            mcq_data_list.append(mcq_data)
        
        # mcq_data_list = mcq_data_list[:10] # test
        now = datetime.now()
        logger = load_logger(log_file_path='log/{}_{}.log'.format(self._llm.name, str(now.date())))
        logger.info('LLM:{}, DataSet:{}, Task type:{}. Experiment begins!'.format(self._llm.name, dataset.name,
                                                                        self._experiment_name))

        results: List[EvaluationResults] = []
        ave_response_time = 0
        ave_response_length = 0
        for i in range(epoch):

            local_response_time = 0 # Local average response time
            local_response_length = 0 # Local average response length
            experimental_results = []

            for j in tqdm.tqdm(range(len(mcq_data_list)), position=0,
                                    desc='LLM:{};experiment:{};epoch:{}/{}'.format(self._llm.name,
                                                                                   self._experiment_name, i, epoch)):
                data = mcq_data_list[j]
                context = data.context
                target = data.target
                option_str = data.options_str
                prompt = self._prompt_generator.generate_prompt(Context=context, TargetWord=target,
                                                                   Options=option_str,
                                                                   Output='')

                if self._prompt_formatter is not None: # Format
                    prompt = self._prompt_formatter.control(prompt)
                
                start = time.time()
                response = self._llm.chat(prompt) # Model output
                end = time.time()
                local_response_time += end - start
                if response is not None:
                    local_response_length += len(response)

                real_set = data.correct_option_index
                real = ''
                for real_index in real_set:
                    real += str(real_index + 1)
                    real += ' '
                real = real.rstrip()
                experimental_result = ExperimentalResult(data, real=real, pre=response)
                experimental_results.append(experimental_result)

            local_response_length /= len(mcq_data_list)
            local_response_time /= len(mcq_data_list)
            ave_response_time += local_response_time
            ave_response_length += local_response_length
            if is_save_record:
                save_record(self, dataset.name, experimental_results, str(i+1))
            local_results = self._evaluator.evaluate(experimental_results, '\d+')
            results.append(local_results)

        ave_response_time = ave_response_time / epoch
        ave_response_length = ave_response_length / epoch
        ave_result = results[0] if len(results) == 1 else merge_evaluationresults(results)

        logger.info('{}-{}:{};average response time:{};average response length:{}'
                    .format(self._llm.name, self._experiment_name,ave_result,
                            ave_response_time, ave_response_length))

        return experimental_results, ave_result
    pass


class GenerationExperiment(LlmExperimenter):
    """ Generation Experimenter

    Attributes:
        experiment_name(str): experiment name
    Methods:
        run_experiment(self, dataset: DataSetInterface, epoch: int=1): Run the generation experiment
    """

    def run_experiment(self, dataset: DatasetInterface, epoch: int=1, is_save_record=False) -> tuple[List[ExperimentalResult], EvaluationResults]|None:
        
        self._prompt_generator.add_prompt_fragment(self._context)
        self._prompt_generator.add_prompt_fragment(self._target_word)
        self._prompt_generator.add_prompt_fragment(self._output)
        

        experimental_results: List[ExperimentalResult] = []
        data_list = dataset.data_list

        now = datetime.now()
        logger = load_logger(log_file_path='log/{}_{}.log'.format(self._llm.name, str(now.date())))
        logger.info('LLM:{}, DataSet:{}, Task type:{}. Experiment begins! '.format(self._llm.name, dataset.name,
                                                                        self._experiment_name))
        
        results: List[EvaluationResults] = []
        ave_response_time = 0
        ave_response_length = 0

        # data_list = data_list[:10] # test

        for i in range(epoch):

            local_response_time = 0 # Local average response time
            local_response_length = 0 # Local average response length
            experimental_results = []

            for j in tqdm.tqdm(range(len(data_list)), position=0,
                                    desc='LLM:{};experiment:{};epoch:{}/{}'.format(self._llm.name,
                                                                                   self._experiment_name, i, epoch)):
                data = data_list[j]
                context = data.context
                target = data.target
                prompt = self._prompt_generator.generate_prompt(Context=context, TargetWord=target,
                                                                   Output='')
                if self._prompt_formatter is not None: # Format
                    prompt = self._prompt_formatter.control(prompt)
                
                start = time.time()
                response = self._llm.chat(prompt) # Model output
                end = time.time()
                local_response_time += end - start
                if response is not None:
                    local_response_length += len(response)

                reals = data._correct_definitions_in_context
                real = ''
                for r in reals:
                    real += r
                    real +='\n'
                real.rstrip('\n')
                experimental_result = ExperimentalResult(data, real=real, pre=response)
                experimental_results.append(experimental_result)
            if is_save_record:
                save_record(self, dataset.name, experimental_results, str(i+1))
            local_response_length /= len(data_list)
            local_response_time /= len(data_list)
            ave_response_time += local_response_time
            ave_response_length += local_response_length

            local_results = self._evaluator.evaluate(experimental_results)
            results.append(local_results)
        
        ave_response_time = ave_response_time / epoch
        ave_response_length = ave_response_length / epoch
        ave_result = results[0] if len(results) == 1 else merge_evaluationresults(results)

        logger.info('{}-{}:{};average response time:{};average response length:{}'
                    .format(self._llm.name, self._experiment_name,ave_result,
                            ave_response_time, ave_response_length))

        return experimental_results, ave_result
    pass


class ContaminationTraceExperimenter(LlmExperimenter):
    """ Contamination Trace Experimenter

    Attributes:
        experiment_name(str): experiment name
        __is_guided(bool): Whether to use guided prompts
    Methods:
        run_experiment(self, dataset: DataSetInterface, epoch: int=1): Run the contamination trace experiment
    """
    def __init__(self, experiment_name, is_guided: bool=False):
        super().__init__(experiment_name)
        self.__is_guided: bool = is_guided
    def run_experiment(self, dataset: DatasetInterface, epoch: int = 1, is_save_record=False):
        data_name = dataset.name
        data_list = dataset.data_list
        data_list = random.sample(data_list, 10)
        if self.__is_guided:
            self._prompt_generator.add_prompt_fragment(GuidedInstructionRequirementEn(data_name))
        else: # not self.__is_guided
            self._prompt_generator.add_prompt_fragment(GenernalInstructionRequirementEn())
        self._prompt_generator.add_prompt_fragment(LeadingWordAndLeadedContent('Label'))
        self._prompt_generator.add_prompt_fragment(LeadingWordAndLeadedContent('First Piece'))
        self._prompt_generator.add_prompt_fragment(LeadingWordAndLeadedContent('Second Piece'))
        
        now = datetime.now()
        logger = load_logger(log_file_path='log/{}_{}.log'.format(self._llm.name, str(now.date())))
        logger.info('LLM:{}, DataSet:{}, Task type:{}. Experiment begins!'.format(self._llm.name, dataset.name,
                                                                        self._experiment_name))

        experimental_results: List[ExperimentalResult] = []
        evaluation_results: List[EvaluationResults] = []
        for i in range(epoch):
            experimental_results = []
            for j in tqdm.tqdm(range(len(data_list)), position=0,
                                    desc='LLM:{};experiment:{};epoch:{}/{}'.format(self._llm.name,
                                                                                   self._experiment_name, i, epoch)):
                data = data_list[j]
                context = data.context
                label = data.correct_definition_in_contexts
                split_index = len(context) // 2
                first_piece = context[:split_index]
                second_piece = context[split_index:]
                prompt = self._prompt_generator.generate_prompt(Label=label, First_Piece=first_piece, Second_Piece='')
                response = self._llm.chat(prompt)

                experimental_result = ExperimentalResult(data, response, second_piece)
                experimental_results.append(experimental_result)
            if is_save_record:
                save_record(self, dataset.name, experimental_results, str(i+1))    
            local_results = self.evaluator.evaluate(experimental_results)
            evaluation_results.append(local_results)
        ave_result = evaluation_results[0] if len(evaluation_results) == 1 else merge_evaluationresults(evaluation_results)
        logger.info('{}-{}:{}'
                    .format(self._llm.name, self._experiment_name,ave_result))

        return experimental_results, ave_result
    pass


class SelfCheckExperimenter(LlmExperimenter):
    """ Self-Check Experimenter

    Attributes:
        experiment_name(str): experiment name
    Methods:
        run_experiment(self, dataset: DataSetInterface, epoch: int=1): Run the self-check experiment
    """

    def run_experiment(self, dataset: DatasetInterface, epoch: int = 1, is_save_record=False):
        data_list = dataset.data_list

        self._prompt_generator.add_prompt_fragment(self._context)
        self._prompt_generator.add_prompt_fragment(self._target_word)
        self._prompt_generator.add_prompt_fragment(self._output)

        now = datetime.now()
        logger = load_logger(log_file_path='log/{}_{}.log'.format(self._llm.name, str(now.date())))
        logger.info('LLM:{}, DataSet:{}, Task type:{}. Experiment begins!'.format(self._llm.name, dataset.name,
                                                                        self._experiment_name))
        # data_list = data_list[:10] # test

        experimental_results: List[ExperimentalResult] = []
        evaluation_results: List[EvaluationResults] = []
        for i in range(epoch):
            local_experimental_results = []
            for j in tqdm.tqdm(range(len(data_list)), position=0,
                                        desc='LLM:{};experiment:{};epoch:{}/{}'.format(self._llm.name,
                                                                                    self._experiment_name, i, epoch)):
                data = data_list[j]
                context = data.context
                target = data.target
                prompt = self._prompt_generator.generate_prompt(Context=context, TargetWord=target, Output='')

                response = self._llm.chat(prompt)
                experimental_result = ExperimentalResult(data, response)
                local_experimental_results.append(experimental_result)
            if is_save_record:
                save_record(self, dataset.name, local_experimental_results, str(i+1))
            local_results = self.evaluator.evaluate(local_experimental_results)
            evaluation_results.append(local_results)
            experimental_results.extend(local_experimental_results)
        ave_result = evaluation_results[0] if len(evaluation_results) == 1 else merge_evaluationresults(evaluation_results)
        logger.info('{}-{}:{}'
                    .format(self._llm.name, self._experiment_name,ave_result))
        return experimental_results, ave_result

