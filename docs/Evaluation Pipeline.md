# Custom Evaluation Pipeline

The `LlmExperimentExecutor` class serves as the core component for configuring and executing customized evaluation pipelines with large language models (LLMs). This module provides flexible interfaces to tailor every aspect of the evaluation process.

#### Core Components

**LlmExperimentExecutor Class**:

- **Attributes**:
  - `llm` (`LargeLanguageModel`): The large language model instance for evaluation
  - `dataset` (`List[DataSetInterface]`): List of datasets used in experiments
  - `epoch` (`int`): Number of experimental iterations
  - `prompt_generator` (`PromptGenerator`): Component for generating LLM prompts
  - `prompt_formatter` (`FormatterController`): Controller for prompt formatting options
  - `_experimenter_list` (`List[LlmExperimenter]`): List of registered experimenters
- **Key Methods**:
  - `add_experimenter(experimenter: LlmExperimenter, evaluator: EvaluatorInterface)`: Registers an experimenter with its corresponding evaluator
  - `execute(is_save_result=False, is_save_record=False)`: Executes all configured experiments with optional result saving

#### Configuration via LlmExecutorConfig

The `LlmExecutorConfig` class provides a streamlined interface for building customized executors:



```python
# Initialize configuration
config = LlmExecutorConfig()

# Set LLM with custom interaction function
config.set_llm('your-model-name', 
               interact_function=custom_interaction_function,
               additional_paras_for_interaction_function )

# Configure datasets (multiple options available)
config.set_dataset_name(['custom-dataset-1', 'custom-dataset-2'])
config.set_data_path_list([['path/to/dataset1'], ['path/to/dataset2']])
# OR use predefined datasets ['semeval2007', 'semeval2013', 'semeval2015', 'senseval2', 'senseval3']
config.choose_default_dataset(0, 2)  # Select first two default datasets
# OR use custom dataset
config.set_custom_dataset(custom_dataset)

# Set prompt generation strategy
config.set_prompt_generator(custom_prompt_fragments)

# Configure prompt formatting
config.set_prompt_formatter(is_title_case=True,
                           is_remove_punctuation=False)

# Set experimental parameters
config.set_epoch(3)  # Three evaluation rounds

# Build the executor
executor = config.get_executor()
```

#### Customization Guide

1. **LLM Configuration**:
   - Implement custom interaction functions for proprietary or local models
   - Support for various API providers and authentication methods
   - [How to specify and configure LLMs]()
2. **Dataset Integration**:
   - Add custom datasets implementing `DataSetInterface`
   - Support for multiple dataset formats and structures
   - [Explore Dataset interfaces]()
3. **Prompt Engineering**:
   - Create custom prompt fragments using `PromptFragmentInterface`
   - Design task-specific prompt strategies
   - [Understand prompt generator interfaces]()
4. **Experiment Design**:
   - Implement custom experimenters via `LlmExperimenter` interface
   - Design novel evaluation paradigms beyond standard WSD tasks
   - [Learn about experimenter interfaces]()
5. **Evaluation Metrics**:
   - Develop custom evaluators implementing `EvaluatorInterface`
   - Create task-specific scoring mechanisms
   - [Implement custom evaluation metrics]()
6. **Output Management**:
   - Configure result saving formats and locations
   - Customize output streaming and logging
   - [Understand result handling and output streams]()
