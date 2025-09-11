# -*- coding: utf-8 -*-
# @Author: Zly
# option_generator_demo
""" This module defines various option generators for multiple-choice questions (MCQs).
    It utilizes the OptionsGeneratorFactory to create different configurations of option generators,
    which can be used in WSD experiments involving large language models (LLMs).
    Each option generator is configured with specific parameters such as the range of correct option indices
    or the quantity of correct options, allowing for flexible experimentation with MCQ formats.
"""

from utils.word_definition_options_generator import *

factory = OptionsGeneratorFactory(is_random=True)
classification_normal_option_generator = factory.get_options_generator_instance

factory = OptionsGeneratorFactory(is_random=True, correct_option_index_range=(0, 1))
classification_op1_option_generator = factory.get_options_generator_instance

factory = OptionsGeneratorFactory(is_random=True, correct_option_index_range=(0, 2))
classification_op2_option_generator = factory.get_options_generator_instance

factory = OptionsGeneratorFactory(is_random=True, correct_option_index_range=(0, 3))
classification_op3_option_generator = factory.get_options_generator_instance

factory = OptionsGeneratorFactory(is_random=True, correct_option_index_range=(0, 4))
classification_op4_option_generator = factory.get_options_generator_instance

factory = OptionsGeneratorFactory(is_random=True, correct_option_index_range=(0, 6))
classification_op6_option_generator = factory.get_options_generator_instance

factory = OptionsGeneratorFactory(is_random=True, correct_option_index_range=(2, 3))
classification_op23_option_generator = factory.get_options_generator_instance

factory = OptionsGeneratorFactory(is_random=True, correct_option_quantity=2)
classification_correct2_option_generator = factory.get_options_generator_instance

factory = OptionsGeneratorFactory(is_random=True, correct_option_quantity=4)
classification_correct4_option_generator = factory.get_options_generator_instance



