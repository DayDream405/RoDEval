# -*- coding: utf-8 -*-
# @Author: Zly
# prompts
"""This module defines prompt fragments for various tasks in English.
These fragments can be used to construct prompts for multiple-choice questions (MCQ),
self-checking tasks, and text generation tasks.
"""
from typing import List

from utils.en_prompts import *
from utils.prompt_generator import PromptFragmentInterface, LeadingWordAndLeadedContent

mcq_prompt_fragments_en: List[PromptFragmentInterface] = []
mcq_task = WsdMcqDefaultTaskRequirementEn()
mcq_prompt_fragments_en.append(mcq_task)

self_check_fragments_en = []
self_check_task = SelfCheckInstructionRequirementEn()
self_check_fragments_en.append(self_check_task)

generation_prompt_fragments_en: List[PromptFragmentInterface] = []
generator_task = GenerationDefaultTaskRequirementEn()
generation_prompt_fragments_en.append(generator_task)

