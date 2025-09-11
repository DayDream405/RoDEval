# -*- coding: utf-8 -*-
# @Author: Zly
# large_language_model
"""Large language model entity class
"""

def default_interact_fun(self, prompt, **kwargs):
    """You need to define a function that accepts a string prompt and a mutable dictionary **kwargs, which is used to store the variables required to call the LLM. \n'
       This function should return the response string returned by the large model receiving the prompt
    """
    raise ValueError('No interact function provided!')


class LargeLanguageModel:
    """Large language model entity class
    This class encapsulates the interaction with large language models, allowing users to define custom interaction functions and manage model-specific parameters.

    Attributes:
        name (str): The name of the large language model, used for saving results.
        interact_function (callable): The function used to interact with the large language model.
        __interact_kwargs (dict): Additional parameters required for interacting with different large language models.
    Methods:
        chat(prompt: str) -> str: Method to interact with the large language model using the provided prompt.
        name (str): Property to get the name of the large language model.
        __str__() -> str: Method to return the string representation of the large language model, which is its name.

    """
    def __init__(self, name:str, interact_function, **kwargs):
        self._name = name
        self._interact_function = interact_function
        self.__interact_kwargs = kwargs

    def chat(self, prompt: str) -> str:
        """Interact with the large language model using the provided prompt.

        Args:
            prompt (str): The prompt to send to the large language model.

        Returns:
            str: The response from the large language model.
        """
        return self._interact_function(prompt, **self.__interact_kwargs)

    @property
    def name(self):
        """Get the name of the large language model."""
        return self._name

    def __str__(self):
        return self._name
