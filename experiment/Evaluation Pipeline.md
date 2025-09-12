# Evaluation Pipeline

一个完整的评测流程需要：

- `entity.large_language_model.LargeLanguageModel`:声明一个被测试的大模型。如何声明
- `utils.prompt_generator.PromptGenerator`:声明一个prompt生成器，用于生成评测任务的prompt。如何声明
- `experiment.experimenter.LlmExperimenter`:声明一个实验器，用于执行一个实验过程。如何声明
- `utils.evaluator.EvaluatorInterface`:声明一个评估器，用于评估模型的答案的得分。如何声明

最后实例化一个`experiment.executor.LlmExperimentExecutor`用于设置上述组件。

你还可以选择使用`experiment.executor.LlmExecutorConfig`类更加快速地完成一个`executor`的设置：

```python
''' To evaluate the performance of GPT-4o on traditional multiple-choice Word Sense Disambiguation (WSD) tasks using OpenAI's API, experiments were conducted over five independent rounds. The test sets consisted of five public English WSD benchmarks: SemEval-2007, SemEval-2013, SemEval-2015, Senseval-2, and Senseval-3.
'''
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

