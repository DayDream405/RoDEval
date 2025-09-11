# -*- coding: utf-8 -*-
# @Author: Zly
# evaluator
"""Evaluation module
    This module provides various evaluators for assessing the performance of LLMs on different WSD tasks.
    Evaluators include:
    - MCQEvaluator: Evaluates multiple-choice question answering for WSD.
    - SelfCheckEvaluator: Evaluates the self-checking porformance of LLMs on WSD tasks.
    - GenerationScoreEvaluator: Evaluates the quality of generated definitions.
    - ContaminationTraceEvaluator: Evaluates potential data contamination in DataSets.    


"""
from typing import List, Dict
from abc import ABC, abstractmethod
import re
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

from entity.results import ExperimentalResult, EvaluationResults, MemoryProbeResults
from utils.tools import *

class EvaluatorInterface(ABC):
    @abstractmethod
    def evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str=None) -> EvaluationResults:
        """Abstract method to evaluate experimental results.

        Args:
            experimental_results: List of experimental results
            answer_extract_pattern(str): Pattern to extract answers (optional)

        Returns:
            EvaluationResults: Evaluation results

        """
        pass
    pass


class MCQEvaluator(EvaluatorInterface):
    """MCQEvaluator
         Calculate P R F1; Accuracy of each word sense; Option recall

       Methods:
           evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str=None) -> EvaluationResults: Calculate P R F1; Accuracy of each word sense; Option recall and return an EvaluationResults instance containing these metrics.

    """
    def evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str=None) -> EvaluationResults:
        """Calculate P R F1; Accuracy of each word sense; Option recall
           

        Args:
            experimental_results: List of experimental results
            answer_extract_pattern(str): Pattern to extract answers (optional)
        Returns:
            EvaluationResults: Evaluation results

        """
        results = EvaluationResults()

        data_totality = len(experimental_results) # total data volume
        sense_totality = 0 # total number of word senses
        key_number_dict: Dict[str, int] = {} # The quantity of each word sense
        options_number_dict: Dict[str, int] = {} # The quantity of each option
        for item in experimental_results:
            sense_keys = item.correct_definition_keys
            for sense_key in sense_keys:
                if sense_key in key_number_dict:
                    key_number_dict[sense_key] += 1
                else: # sense_key not in key_number_dict
                    key_number_dict[sense_key] = 1
                    sense_totality += 1

            option_number = item.real
            if option_number in options_number_dict:
                options_number_dict[option_number] += 1
            else: # option_number not in options_number_dict
                options_number_dict[option_number] = 1
        correct_answer_key_number_dict: Dict[str, int] = dict.fromkeys(key_number_dict.keys(), 0)

        local_recall = 0
        acc_count = 0  # accuracy number
        true_positive_option_number_dict: Dict[str, int] = dict.fromkeys(options_number_dict.keys(), 0) # True positive number of each option
        for item in experimental_results:
            keys = item.correct_definition_keys
            pre: str = item.pre # Model predicts answers
            if pre is None:
                pre = ''
            real: str = item.real # True answer
            is_correct = False # Whether the prediction is correct

            if len(pre.rstrip()) == 0: # Skip empty prediction(Refer to the previous WSD algorithm of F1)
                continue

            if answer_extract_pattern is not None: # Use regex to extract answer
                matches = re.findall(answer_extract_pattern, pre)
                if len(matches) != 0:
                    pre = matches[0]
            # Loose matching
            if real.find(pre) != -1:
                is_correct = True

            if is_correct:
                acc_count += 1
                for key in keys:
                    correct_answer_key_number_dict[key] = correct_answer_key_number_dict[key] + 1
                    local_recall += 1 / key_number_dict[key]
                true_positive_option_number_dict[real] += 1
        p = acc_count / data_totality
        r = local_recall / sense_totality
        f = 2 * p * r / (p + r) if p + r != 0 else 0

        sense_accuracy: Dict[str, float] = {}
        for k, v in correct_answer_key_number_dict.items():
            sense_accuracy[k] = v / key_number_dict[k]

        option_recall: Dict[str, float] = {}
        for k, v in options_number_dict.items():
            o_r = true_positive_option_number_dict[k] / v
            option_recall[k] = o_r

        results.add_result('p', p)
        results.add_result('r', r)
        results.add_result('f1', f)
        results.add_result('sense_accuracy', sense_accuracy)
        results.add_result('option_recall', option_recall)

        return results

class SelfCheckEvaluator(EvaluatorInterface):
    """SelfCheckEvaluator
         Calculate the disambiguation accuracy of self-checking

       Methods:
           evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str = None) -> EvaluationResults: Calculate the disambiguation accuracy of self-checking and return an EvaluationResults instance containing this metric.

    """
    def evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str = None) -> EvaluationResults:
        results = EvaluationResults()
        quantity = len(experimental_results)
        positive_check = 0 # Number of positive outputs
        real = 'yes' # The positive answer for self-checking
        for item in experimental_results:
            pre = item.pre.lower()
            is_positive = False

            if answer_extract_pattern is not None:
                matches = re.findall(answer_extract_pattern, pre)
                if len(matches) != 0:
                    pre = matches[0]

            if answer_extract_pattern is not None:
                if pre == real:
                    is_positive = True
            else: # answer_extract_pattern is None
                if pre.find(real) != -1:
                    is_positive = True

            if is_positive:
                positive_check += 1

        results.add_result('self_check', (positive_check / quantity))
        return results


class TraditionalGenerationCriteriaEvaluator(EvaluatorInterface):
    def evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str = None) -> EvaluationResults:
        """Calculate traditional generation criteria: Loose Matching, BLEU, ROUGE

        Args:
            experimental_results: List of experimental results
            answer_extract_pattern: Pattern to extract answers (optional)

        Returns:
            EvaluationResults: Evaluation results

        """
        results = EvaluationResults()
        data_quantity = len(experimental_results)
        ok = notok = 0 # Variables of loose matching
        relaxed_matching_acc: Dict[str, List[float]] = {}
        ave_rm_f1 = 0
        bleu: Dict[str, List[float]] = {}
        ave_bleu = 0
        rouge_1: Dict[str, List[float]] = {}
        rouge_2: Dict[str, List[float]] = {}
        rouge_l: Dict[str, List[float]] = {}
        ave_rouge_1= ave_rouge_2 = ave_rouge_l = 0

        for item in experimental_results:
            real = item.real
            reals = real.split('\n')
            pre = item.pre
            name = item.polysemous.name
            if pre is None:
                pre = ''
            if answer_extract_pattern is not None:
                matches = re.findall(answer_extract_pattern, pre)
                if len(matches) != 0:
                    pre = matches[0]

            # Loose matching
            word_list: List[str] = pre.split(' ')
            local_ok = local_notok = 0 # Intermediate variables of loose matching
            for r in reals:
                max_local_ok = min_local_notok = 0
                for w in word_list:
                    if w in r:
                        max_local_ok += 1
                    else: # w not in r
                        min_local_notok += 1
                if max_local_ok >= local_ok:
                    local_ok = max_local_ok
                    local_notok = min_local_notok
            ok += local_ok / len(word_list)
            notok += local_notok / len(word_list)
            local_rm_f1 = local_ok / (local_ok + local_notok)
            ave_rm_f1 += local_rm_f1
            if name in relaxed_matching_acc.keys():
                relaxed_matching_acc[name].append(local_rm_f1)
            else: # name not in relaxed_matching_acc.keys()
                relaxed_matching_acc[name] = [local_rm_f1]

            # BLEU
            pre_tokens: List[str] = word_list
            bleu_score = 0
            for r in reals:
                real_tokens: List[str] = r.split(' ')
                smoothing = SmoothingFunction()
                step_score = sentence_bleu(pre_tokens, real_tokens, smoothing_function=smoothing.method1)
                bleu_score = step_score if step_score > bleu_score else bleu_score
            ave_bleu += bleu_score
            if name in bleu.keys():
                bleu[name].append(bleu_score)
            else: # sense_key not in bleu.keys()
                bleu[name] = [bleu_score]
            
            # ROUGE
            r1=r2=rl=0
            if pre != '' and pre is not None:
                for r in reals:
                    rouge_instance = Rouge()
                    rouge_scores = rouge_instance.get_scores(pre, real)
                    step_r1 = rouge_scores[0]['rouge-1']['f']
                    step_r2 = rouge_scores[0]['rouge-2']['f']
                    step_rl = rouge_scores[0]['rouge-l']['f']
                    r1 = step_r1 if step_rl > r1 else r1
                    r2 = step_r2 if step_r2 > r2 else r2
                    rl = step_rl if step_rl > rl else rl
                    
            
            ave_rouge_1 += r1
            ave_rouge_2 += r2
            ave_rouge_l += rl

            if name in rouge_1.keys():
                rouge_1[name].append(r1)
                rouge_2[name].append(r2)
                rouge_l[name].append(rl)
            else:
                rouge_1[name] = [r1]
                rouge_2[name] = [r2]
                rouge_l[name] = [rl]

        ave_rm_f1 /= data_quantity
        ave_bleu /= data_quantity
        ave_rouge_1 /= data_quantity
        ave_rouge_2 /= data_quantity
        ave_rouge_l /= data_quantity

        sense_rm: Dict[str, float] = {}
        sense_bleu: Dict[str, float] = {}
        sense_rouge_1: Dict[str, float] = {}
        sense_rouge_2: Dict[str, float] = {}
        sense_rouge_l: Dict[str, float] = {}
        for k in relaxed_matching_acc.keys():
            rms = relaxed_matching_acc[k]
            bleus = bleu[k]
            rouges_1 = rouge_1[k]
            rouges_2 = rouge_2[k]
            rouges_l = rouge_l[k]
            sense_rm[k] = sum(rms) / len(rms)
            sense_bleu[k] = sum(bleus) / len(bleus)
            sense_rouge_1[k] = sum(rouges_1) / len(rouges_1)
            sense_rouge_2[k] = sum(rouges_2) / len(rouges_2)
            sense_rouge_l[k] = sum(rouges_l) / len(rouges_l)
        

        results.add_result('rm', ave_rm_f1) # Loose matching accuracy
        results.add_result('sense_rm', sense_rm) # Each data's loose matching accuracy
        results.add_result('bleu', ave_bleu) # BLEU
        results.add_result('sense_bleu', sense_bleu) # Each data's BLEU
        results.add_result('rouge_1', ave_rouge_1)
        results.add_result('rouge_2', ave_rouge_2)
        results.add_result('rouge_l', ave_rouge_l)
        results.add_result('sense_rouge_1', sense_rouge_1) # Each data's ROUGE-1
        results.add_result('sense_rouge_2', sense_rouge_2)
        results.add_result('sense_rouge_l', sense_rouge_l)

        return results


class GenerationScoreEvaluator(EvaluatorInterface):
    """GenerationScoreEvaluator
            Calculate the generation score of predicted definitions

       Methods:
           evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str = None) -> EvaluationResults: Calculate the generation score of predicted definitions and return an EvaluationResults instance containing this metric.

    """
    def __init__(self):
        self._is_use_llm_embedding = True
    @property
    def is_use_llm_embedding(self) -> bool:
        return self._is_use_llm_embedding
    @is_use_llm_embedding.setter
    def is_use_llm_embedding(self, value: bool):
        self._is_use_llm_embedding = value
    
    def evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str = None) -> EvaluationResults:
        results = EvaluationResults()
        data_quantity = len(experimental_results)
        word_frequency: Dict[str, int] = {}

        sim_list = []
        for item in experimental_results:
            pre = item.pre
            real = item.real
            reals = real.split('\n')
            if pre is None:
                pre = ''
            if answer_extract_pattern is not None:
                matches = re.findall(answer_extract_pattern, pre)
                if len(matches) != 0:
                    pre = matches[0]

            pre_words: List[str] = remove_punctuation(pre).split(' ')
            for w in pre_words: # Count word frequency
                if w in word_frequency:
                    word_frequency[w] += 1
                else: # w not in word_frequency
                    word_frequency[w] = 1
            max_sim = 0
            for r in reals:

                if self.is_use_llm_embedding:
                    step_sim = sentence_similarity_llm(pre, r) # LLM embedding
                else:
                    step_sim = sentence_similarity(pre, real) #BERT embedding

                max_sim = step_sim if step_sim > max_sim else max_sim
            sim_list.append(max_sim)

        # Calculate word scores based on frequency
        max_frequency = 0
        word_score = {}
        for f in word_frequency.values():
            max_frequency = f if f > max_frequency else max_frequency
        for w in word_frequency.keys():
            count = word_frequency[w]
            dif = max_frequency - count
            word_score[w] = dif / (max_frequency / 100)

        generate_score: float = 0
        sense_scores: Dict[str, List[float]] = {}

        for i in range(data_quantity):
            item = experimental_results[i]
            similarity_score = sim_list[i]
            pre = item.pre
            if pre is not None:
                if answer_extract_pattern is not None:
                    matches = re.findall(answer_extract_pattern, pre)
                    if len(matches) != 0:
                        pre = matches[0]
                pre_words: List[str] = remove_punctuation(pre).split(' ')
                local_richness_score = 0
                for w in pre_words:
                    local_richness_score += word_score[w]
                local_richness_score /= len(pre_words) * 100
                # Generation score formula
                local_generate_score = similarity_score + ((1 - similarity_score) * local_richness_score * similarity_score * similarity_score)
            else: # pre is None
                local_generate_score = 0           
            generate_score += local_generate_score
            data_name = item.polysemous.name
            if data_name not in sense_scores:
                scores_list = [local_generate_score]
                sense_scores[data_name] = scores_list
            else: # data_name in sense_scores
                sense_scores[data_name].append(local_generate_score)
        generate_score /= data_quantity
        ave_sense_score: Dict[str, float] = dict.fromkeys(sense_scores.keys(), 0)
        for k, score_list in sense_scores.items():
            ave_sense_score[k] = (sum(score_list) / len(score_list))

        results.add_result('generate_score', generate_score)
        results.add_result('sense_score', ave_sense_score)
        
        return results
    pass


class ContaminationTraceEvaluator(EvaluatorInterface):
    """ContaminationTraceEvaluator
            Calculate BLEU and ROUGE-L to trace potential data contamination
            Reference to "TIME TRAVEL IN LLMS: TRACING DATA  CONTAMINATION IN LARGE LANGUAGE MODELS"(ICLR 2024)
         Methods:
              evaluate(self, experimental_results: List[ExperimentalResult],
                  answer_extract_pattern: str = None) -> EvaluationResults: Calculate BLEU and ROUGE-L to trace potential data contamination and return an EvaluationResults instance containing these metrics.

    """

    def evaluate(self, experimental_results: List[ExperimentalResult],
                 answer_extract_pattern: str=None) -> EvaluationResults:
        results = EvaluationResults()
        data_quantity = len(experimental_results)
        bleu = 0
        rouge_l = 0
        for data in experimental_results:
            
            pre = data.pre
            if pre is None:
                pre = ''
            real = data.real
            pre_tokens = pre.split()
            real_tokens = real.split()
            local_bleu = sentence_bleu(pre_tokens, real_tokens)
            bleu += local_bleu

            rouge_instance = Rouge()
            rouge_scores = rouge_instance.get_scores(pre, real)
            local_rougle_l = rouge_scores[0]['rouge-l']['f']
            rouge_l+= local_rougle_l

        bleu = bleu / data_quantity
        rouge_l = rouge_l / data_quantity
        results.add_result('bleu', bleu)
        results.add_result('rouge_l', rouge_l)

        return results
        
