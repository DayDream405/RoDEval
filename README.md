# RoDEval: A Robust Word Sense Disambiguation Evaluation Framework for Large Language Models (EMNLP2025 main)

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. Interface Definition (`Abstract Base Class`)
4. Predefined Implementations
6. Detailed Description
   - 5.1. `method_name`
   - 5.2. `property_name`

## 1. Overview

RoDEval is a specialized evaluation framework designed for assessing the Word Sense Disambiguation (WSD) capabilities of large language models (LLMs). It provides researchers with a flexible and extensible toolkit to measure model performance rigorously and systematically.

**Key Features:**

- **Customizable Evaluation:** Supports user-provided test sets, enabling evaluation on proprietary or domain-specific data.
- **Extensible Task Design:** Allows for the creation of custom WSD task formulations beyond predefined paradigms.
- **Adaptable Metrics:** Facilitates the implementation of novel evaluation methodologies tailored to specific research needs.
- **Unified Data Handling:** Standardizes data formats and processing pipelines, ensuring consistency and reproducibility across experiments.
- **Built-in Benchmarks:** Comes pre-equipped with data processors for five standard English WSD datasets: SemEval-2007, SemEval-2013, SemEval-2015, Senseval-2, and Senseval-3.
- **Diverse Task Library:** Includes three predefined WSD task types (e.g., multiple-choice QA, definition generation) and a dedicated data contamination detection task.

RoDEval is designed to streamline the benchmarking process, offering both out-of-the-box functionality for standard evaluations and the flexibility needed for cutting-edge research in WSD.

## 2. Quick Start

This guide will help you set up a complete Word Sense Disambiguation (WSD) evaluation pipeline and start assessing LLMs' disambiguation capabilities.

**1. Installation**
First, install the required dependencies:

```python
pip install -r requirements.txt
```

**Note:** You must separately configure the runtime environment for any LLMs you intend to test.



------

**2. Basic Setup**
The following code demonstrates how to configure and execute a standard WSD evaluation pipeline:

```python
from experiment.executor import LlmExecutorConfig
from demo.prompts import *
from demo.experimenter_demo import *
from utils.evaluator import *
from openai import OpenAI

# Initialize LLM client
client = OpenAI(
    api_key='your-api-key-here',  # Replace with your actual API key
    base_url='your-base-url-here'  # Replace with your actual API endpoint
)

def interaction_function(prompt, **kwargs):
    """Basic LLM interaction function."""
    completion = kwargs['client'].chat.completions.create(
        model="deepseek-chat",  # Model identifier
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ],
        stream=False,
    )
    return completion.choices[0].message.content

# Configure the evaluation pipeline
executor_config = LlmExecutorConfig()
executor_config.set_llm('gpt-4o', interact_function=interaction_function, client=client)
executor_config.set_epoch(5)  # Set number of evaluation rounds
executor_config.choose_default_dataset(0, 5)  # Use all 5 default datasets
executor_config.set_prompt_generator(mcq_prompt_fragments_en)  # Set prompt generator

# Configure evaluation components
experimenter = classification_normal_experimenter
evaluator = MCQEvaluator()  # Multiple-choice question evaluator as an example
executor_config.add_experimenter(experimenter, evaluator)

# Initialize executor
executor = executor_config.get_executor()
```



------

**3. Execution**
Run the evaluation pipeline with results saving enabled:

```python
# Execute evaluation
executor.execute(is_save_record=True,  # Save detailed interaction records
                 is_save_result=True)   # Save final evaluation results
```



------

**Next Steps**

- See [Custom Evaluation](docs\Evaluation Pipeline.md) for advanced configuration options
- Learn about [Custom Metrics]() for implementing custom evaluation protocols
- Explore [Dataset Integration]() for adding new evaluation datasets

For detailed API documentation, visit our [API Reference](https://link-to-docs/).

你可以参考以下代码快速设置一个完整的WSD评测pipeline，并开始评测大模型的词义消歧能力

This framework includes pre-built modules for processing five public English word sense disambiguation (WSD) test datasets, three types of WSD tasks, and one data leakage detection task.

- Click [here](docs/Evaluation%20Pipeline.md) to see how to use the predefined methods to construct a complete evaluation pipeline.

The framework also supports custom evaluation:

- To view the interfaces supported by the framework, click [here].
- For instructions on using other datasets, click [here].
- For guidance on setting up large language models, click [here].
- To learn how to customize experiments, click [here].
- For details on customizing an evaluation method, click [here].
- For a tutorial on building a fully custom evaluation pipeline, click [here].

For any questions, please contact us: luyang.zhang.qlu@gmail.com

