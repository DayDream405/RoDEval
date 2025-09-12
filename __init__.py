# -*- coding: utf-8 -*-
# @Author: Zly
# demo
from openai import OpenAI
import numpy as np

from demo.experimenter_demo import *
from entity.large_language_model import LargeLanguageModel
from experiment.executor import LlmExecutorConfig
from experiment.experimenter import McqExperimenter
from utils.evaluator import MCQEvaluator, GenerationScoreEvaluator, ContaminationTraceEvaluator, SelfCheckEvaluator
from demo.prompts import *
from utils.wordnet_tools import *

"""To evaluate the performance of GPT-4o on traditional multiple-choice Word Sense Disambiguation (WSD) tasks using OpenAI's API, 
    experiments were conducted over five independent rounds. 
   The test sets consisted of five public English WSD benchmarks: SemEval-2007, SemEval-2013, SemEval-2015, Senseval-2, and Senseval-3.
"""

client = OpenAI(
    api_key='api-key',  # replace with your actual API key
    base_url='api.url'  # replace with your actual base URL
)

def fun(prompt, **kwargs):
    """Interaction function for the LLM."""
    completion = kwargs['client'].chat.completions.create(
        model="deepseek-chat",  # this field is currently unused
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user",
             "content": prompt}
        ],
        stream=False,
    )
    return completion.choices[0].message.content

def main():
    executor_fig = LlmExecutorConfig()
    # set llm
    executor_fig.set_llm('gpt-4o', interact_function=fun, client=client)
    # set epoch
    executor_fig.set_epoch(5)
    # choose dataset
    # executor_fig.choose_default_dataset(0, 1) # defult (0, 5)
    # set prompt generator
    executor_fig.set_prompt_generator(mcq_prompt_fragments_en)
    # set experimenter and evaluator
    experimenter = classification_normal_experimenter
    evaluator = MCQEvaluator()
    executor_fig.add_experimenter(experimenter, evaluator)
    # get executor
    executor = executor_fig.get_executor()
    # run
    executor.execute(is_save_record=True, is_save_result=True)
    pass




if __name__ == '__main__':
    main()