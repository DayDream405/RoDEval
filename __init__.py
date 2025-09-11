# -*- coding: utf-8 -*-
# @Author: Zly
# demo
from openai import OpenAI
import numpy as np

from demo.experimenter_demo import *
from entity.large_language_model import LargeLanguageModel
from experiment.experiment_executor import LlmExecutorConfig
from experiment.experimenter import McqExperimenter
from utils.evaluator import MCQEvaluator, GenerationScoreEvaluator, ContaminationTraceEvaluator, SelfCheckEvaluator
from demo.prompts import *
from utils.wordnet_tools import *

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
    # executor_fig.choose_default_dataset(0, 1)
    # set prompt generator
    executor_fig.set_prompt_generator(mcq_prompt_fragments_en)
    executor = executor_fig.get_executor()
    # set experimenter and evaluator
    experimenter = classification_normal_experimenter
    evaluator = MCQEvaluator()
    executor.add_experimenter(experimenter, evaluator)
    # run
    executor.execute(is_save_record=True, is_save_result=True)
    pass




if __name__ == '__main__':
    main()