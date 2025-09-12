# Evaluation Pipeline

A complete evaluation pipeline requires the following components:

- `entity.large_language_model.LargeLanguageModel`:Declare a large language model to be tested. [How to declare]
- `utils.prompt_generator.PromptGenerator`:Declare a prompt generator used to create prompts for the evaluation tasks. [How to declare]
- `experiment.experimenter.LlmExperimenter`:Declare an experimenter responsible for executing the experimental process. [How to declare]
- `utils.evaluator.EvaluatorInterface`:Declare an evaluator used to score the model's answers. [How to declare]

Finally, instantiate an `experiment.executor.LlmExperimentExecutor`to configure the components mentioned above.

Optionally, you can use the`experiment.executor.LlmExecutorConfig` class to set up an `executor` more rapidly.

```python
 """
    To evaluate the performance of GPT-4o on traditional multiple-choice 
    Word Sense Disambiguation (WSD) tasks using OpenAI's API, experiments 
    were conducted over five independent rounds. The test sets consisted 
    of five public English WSD benchmarks: SemEval-2007, SemEval-2013, 
    SemEval-2015, Senseval-2, and Senseval-3.
"""
from experiment.executor import LlmExecutorConfig
from demo.prompts import *
from demo.experimenter_demo import *
from utils.evaluator import MCQEvaluator

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
	
executor_fig = LlmExecutorConfig()
# set llm
executor_fig.set_llm('gpt-4o', interact_function=fun, client=client)
# set epoch
executor_fig.set_epoch(5)
# choose dataset
executor_fig.choose_default_dataset(0, 5) # defult (0, 5)
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
```

