# -*- coding: utf-8 -*-
# @Author: Zly
# prompt_generator
"""Prompt generation module
This module provides classes and interfaces for generating prompts for large language models (LLMs).
The main components include:
- PromptFragmentInterface: An abstract base class defining the interface for prompt fragments.
- CustomPrompt: A class for adding custom prompt content.
- StaticContentInterface: An interface for static content in prompts, such as task instructions.
- LeadingWordAndLeadedContent: A class for managing leading words and their associated content in prompts.
- PromptGenerator: A class that assembles the final prompt string based on added prompt fragments.
Default prompt format for the framework:
<StaticContent> {LeadingWordAndLeadedContent1} {LeadingWordAndLeadedContent2} ... {LeadingWordAndLeadedContentN}
Example:
<You are a helpful assistant.
Please select the option corresponding to the definition of the TargetWord in the context from the candidate definitions.
You only need to output the serial number corresponding to the definition of the target word. 
Any additional output will reduce the quality of your answer.>
{Context: <context>}{TargetWord: <target_word>}{Definitions: <definitions>}
{Output:}
"""

from abc import ABC, abstractmethod
from typing import List
import warnings

class PromptFragmentInterface(ABC):
    """prompt fragment interface
        Methods:
            generate_prompt(self, content: str) -> str: @abstractmethod Add prompt fragment to prompt

    """

    @abstractmethod
    def generate_prompt(self, content: str) -> str:
        """@abstractmethod Add prompt fragment to prompt

        Args:
            content(str): Current prompt content

        Returns:
            str: prompt fragment\n

        """
        pass

    pass

class CustomPrompt(PromptFragmentInterface):
    """Custom prompt fragment for adding arbitrary content to the prompt.
       This class allows users to add any custom content to the prompt.
    """

    def generate_prompt(self, content: str):
        """

        Returns:
            str: content\n

        """
        return '{}\n'.format(content)


class StaticContentInterface(PromptFragmentInterface):
    """Static content interface for prompts, such as task instructions.
       This interface is used to define static content in prompts that does not change.

        Methods:
            static_content: @property @abstractmethod str: Abstract property for static content
    """

    def __init__(self):
        super().__init__()
        pass

    @property
    @abstractmethod
    def static_content(self):
        """static content abstract property
        """
        pass

    def generate_prompt(self, content: str):
        """

        Returns:
            str: content self.__static_content\n

        """
        return '{}{}\n'.format(content, self.static_content)


class LeadingWordAndLeadedContent(PromptFragmentInterface):
    """Leading word and leaded content class, used to manage leading words and their associated content in prompts.
    The leading word serves as a key for the instance, and the leaded content is the content to be added after the leading word.
    The default format is leading_word: leaded_content\n

       Attributes:
           _leading_word(str): Leading word
           _leaded_content(str): Leaded content, initially empty

    """

    def __init__(self, leading_word: str):
        """
        Args:
            leading_word(str): Leading word
        """

        self._leading_word: str = leading_word
        self._leaded_content: str = ''

    def generate_prompt(self, content: str):
        """
        Returns:
            str: content + leading_word: leaded_content\n

        """

        return '{}{}: {}\n'.format(content, self._leading_word, self._leaded_content)

    def get_leading_word(self) -> str:
        """get leading word

        Returns:
            str: self._leading_word

        """
        return self._leading_word

    def set_leaded_content(self, content: str) -> None:
        """set leaded content

        Args:
            content(str): Leaded content

        Returns:
            None

        """
        self._leaded_content = content
        pass

    def __eq__(self, other) -> bool:
        """

        Returns:
            bool: self._leading_word == other.get_leading_word()

        """
        if isinstance(other, LeadingWordAndLeadedContent):

            return self._leading_word == other.get_leading_word()

        return False

    def __hash__(self) -> int:
        """

        Returns:
            hash(self._leading_word)

        """
        return hash(self._leading_word)


class PromptGenerator(object):
    """Prompt generator class that assembles the final prompt string based on added prompt fragments.
        The default prompt format for the framework is:
        <StaticContent> {LeadingWordAndLeadedContent1} {LeadingWordAndLeadedContent2} ... {LeadingWordAndLeadedContentN}
        Example:
        <You are a helpful assistant.
        Please select the option corresponding to the definition of the TargetWord in the context from the candidate definitions.
        You only need to output the serial number corresponding to the definition of the target word.
        Any additional output will reduce the quality of your answer.>
        {Context: <context>}{TargetWord: <target_word>}{Definitions: <definitions>}
        {Output:}

        Attributes:
            __prompt_fragments(List[PromptFragmentInterface]): List of added PromptFragment instances

        Methods:
            add_prompt_fragment(self, fragment: PromptFragmentInterface) -> None: Add new PromptFragment instance
            remove_fragment(self, fragment: PromptFragmentInterface) -> None: Remove prompt fragment
            generate_prompt(self, **kwargs) -> str: Generate prompt string based on set prompt fragments

    """

    def __init__(self):
        self.__prompt_fragments: List[PromptFragmentInterface] = []
        pass

    def add_prompt_fragment(self, fragment: PromptFragmentInterface) -> None:
        """ Add new PromptFragment instance

        Args:
            fragment(PromptFragmentInterface): New PromptFragment instance to add
        Returns:
            None

        """
        self.__prompt_fragments.append(fragment)
        pass

    def remove_fragment(self, fragment: PromptFragmentInterface) -> None:
        """ Remove prompt fragment

        Args:
            fragment(PromptFragmentInterface): PromptFragment instance to remove

        Returns:
            None

        """
        if fragment in self.__prompt_fragments:
           self.__prompt_fragments.remove(fragment)
        return None

    def generate_prompt(self, **kwargs) -> str:
        """ Generate prompt string based on set prompt fragments

        Args:
            **kwargs: Dictionary of leading words' corresponding content. The keys are leading words with spaces replaced by underscores,
                      and the values are the corresponding content to be filled in.

        Returns:
            str: Generated prompt string

        """
        prompt: str = ''
        for fragment in self.__prompt_fragments:
            if isinstance(fragment, StaticContentInterface):
                prompt = fragment.generate_prompt(prompt)
                pass
            elif isinstance(fragment, LeadingWordAndLeadedContent):
                leading_word = fragment.get_leading_word()
                leading_word = leading_word.replace(' ', '_')
                content = kwargs.get(leading_word, '')
                if content == '':
                    warnings.warn('{}的内容为空'.format(fragment.get_leading_word()), RuntimeWarning)

                fragment.set_leaded_content(content)
                prompt = fragment.generate_prompt(prompt)

        return prompt
    pass




