# RoDEval: A Robust Word Sense Disambiguation Evaluation Framework for Large Language Models (EMNLP2025 main)

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Frequently Asked Questions](#3-Frequently-Asked-Questions)

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

## 3. Frequently Asked Questions

**Q: Is your framework a benchmark?**

**A:** You could consider it a WSD benchmark for large language models, but that isn't entirely the original intent behind proposing this framework. We believe that evaluation methods should evolve alongside the rapid development of large language models. The main motivation for introducing this framework is precisely because a single correctness metric is insufficient to comprehensively assess a model's WSD capability. Therefore, we have considered as many real-world application scenarios as possible and proposed four evaluation metrics.

While it's true that our metrics address certain issues overlooked by traditional benchmarks, they inevitably have their own limitations and cannot cover every possible application scenario. As a result, we do not intend for our metrics to be adopted as a universal standard for comparing scores. Moreover, if certain current WSD issues in large models (such as WSD robustness or model biases) are resolved in the future, some of these metrics may no longer be relevant. Simultaneously, new challenges may emerge in the WSD performance of large models.

Thus, our framework is designed to be extensible and aims to inspire further research. We welcome researchers to test models using their own evaluation criteria.

------

**Q: Why is there no final composite score (e.g., a "WSD Score") to measure the overall WSD capability of models?**

A: Using a single metric to evaluate a specific capability of large language models is inappropriate. Given the complex real-world application scenarios of these models, determining which aspect of their capability should be prioritized is subjective. In fact, we considered deriving an ultimate score to represent a model's WSD ability, but ultimately abandoned the idea based on the above considerations.

------

**If you find our framework helpful for your research, please don't hesitate to give us a star!**

------

**Next Steps**

- See [Custom Evaluation](docs/Evaluation%20Pipeline.md) for advanced configuration options
- Implement [experimenter interface]() for designing new WSD task
- Learn about [Custom Metrics]() for implementing custom evaluation protocols
- Explore [Dataset Integration]() for adding new evaluation datasets

For detailed API documentation, visit our [API Reference](https://link-to-docs/).

------

### Author Says

**Word Sense Disambiguation(WSD)** was initially proposed as a subtask of machine translation and was considered a relatively straightforward NLP task. Today, however, both LLMs and small classification models have achieved excellent performance in machine translation and are widely deployed in practical applications. Yet, the performance of these models on WSD tasks remains limited, or at least difficult to improve substantially. I believes that our understanding of the difficulty of WSD may be inaccurate, or that current models might not truly disambiguate word senses when performing translation tasks. Therefore, further research on WSD could potentially help models learn to genuinely understand complex semantics. This remains a somewhat niche classification task at present, and I hope more researchers will begin to pay attention to this important research direction.

------

###### For any issues, please contact us: luyang.zhang.qlu@gmail.com
###### 临近毕业，疲于找博士导师中。。。我会在学业稳定后再回复问题并更新缺失的文档，请谅解。
###### Approaching graduation, exhausted from searching for a doctoral supervisor... I will reply to issues and update missing documents after my studies have stabilized. Please understand.
