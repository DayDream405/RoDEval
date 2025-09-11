# -*- coding: utf-8 -*-
# @Author: Zly
# experimenter_demo
""" This module defines various experimenter instances for conducting WSD experiments with large language models (LLMs).
    It imports option generators and formatters from other demo modules to create different configurations of experimenters.
    Each experimenter is set up with a specific option generator and, in some cases, a prompt formatter to facilitate diverse experimental setups.
"""

from demo.option_generator_demo import *
from experiment.experimenter import *
from demo.formatter_demo import *

classification_normal_experimenter = McqExperimenter('classification-normal', classification_normal_option_generator)
classification_op1_experimenter = McqExperimenter('classification-op1', classification_op1_option_generator)
classification_op2_experimenter = McqExperimenter('classification-op2', classification_op2_option_generator)
classification_op3_experimenter = McqExperimenter('classification-op3', classification_op3_option_generator)
classification_op4_experimenter = McqExperimenter('classification-op4', classification_op4_option_generator)
classification_op6_experimenter = McqExperimenter('classification-op6', classification_op6_option_generator)
classification_op23_experimenter = McqExperimenter('classification-op2-3', classification_op23_option_generator)


classification_corr2_experimenter = McqExperimenter('classification-corr2', classification_correct2_option_generator)
classification_corr4_experimenter = McqExperimenter('classification-corr4', classification_correct4_option_generator)
classification_op3_experimenter = McqExperimenter('classification-op3', classification_op3_option_generator)

classification_upper_experimenter = McqExperimenter('classification-upper', classification_normal_option_generator)
classification_upper_experimenter._prompt_formatter = uppercase_controller

classification_lower_experimenter = McqExperimenter('classification-lower', classification_normal_option_generator)
classification_lower_experimenter._prompt_formatter = lowercase_controller

classification_title_experimenter = McqExperimenter('classification-title', classification_normal_option_generator)
classification_title_experimenter._prompt_formatter = titlecase_controller

classification_removepunc_experimenter = McqExperimenter('classification-robust_removepunc', classification_normal_option_generator)
classification_removepunc_experimenter._prompt_formatter = remove_punctuation_controller


self_check_experimenter = SelfCheckExperimenter('self check-negative')

generation_experimenter = GenerationExperiment('generation')

contamination_trace_guided_experimenter = ContaminationTraceExperimenter('contamination-guided instruction', is_guided=True) # 数据泄露检测
contamination_trace_general_experimenter = ContaminationTraceExperimenter('contamination-general instruction') # 数据泄露检测
