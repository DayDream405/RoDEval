# -*- coding: utf-8 -*-
# @Author: Zly
# experimental_data
"""Experimental data entity class

"""
from typing import List, Literal, Set

from entity.option_list import MCQOptionsList


class Polysemous(object):
    """Polysemous word class

    Attributes:
        name (str): The string representation of the polysemous word.
        sense_definitions_list (List[str]): List of all definitions of the polysemous word.
    Methods:
        name (str): Property to get the name of the polysemous word.
        sense_definitions_list (List[str]): Property to get the list of definitions of the polysemous word.
        __str__() -> str: Method to return the string representation of the polysemous word, which is its name.
        to_json() -> dict: Method to convert the polysemous word object to a JSON-compatible dictionary.

    """

    def __init__(self, name: str, sense_definitions_list: List[str]):
        self._name: str = name
        self._sense_definitions_list: List[str] = sense_definitions_list

    @property
    def name(self) -> str:
        return self._name

    @property
    def sense_definitions_list(self) -> List[str]:
        return self._sense_definitions_list

    def __str__(self):
        return self._name

    def to_json(self):
        json_dict = {}
        json_dict['name'] = self.name
        json_dict['sense_definitions_list'] = self.sense_definitions_list
        return json_dict
    pass


class ExperimentalData(object):
    """Experimental data entity class
    This class encapsulates the data required for WSD, including the context, target word, and correct definitions.

    Attributes:
        polysemous (Polysemous): The polysemous word associated with the experimental data
        context (str): The context in which the polysemous word appears
        target (str): The target word
        correct_definition_in_context (str): The correct definition of the polysemous word in the given context
        correct_definition_keys (List[str]): The unique key(s) for the correct definition(s)
    Methods:
        definitions_list (List[str]): Property to get the list of definitions of the polysemous
        polysemous (Polysemous): Property to get the polysemous word
        context (str): Property to get the context
        target (str): Property to get the target word
        correct_definition_in_contexts (str): Property to get the correct definition(s) in context
        correct_definition_keys (List[str]): Property to get the unique key(s) for the correct definition(s)
        to_json() -> dict: Method to convert the experimental data object to a JSON-compatible dictionary
        __str__() -> str: Method to return the string representation of the experimental data, including polysemous name, context, and correct definition keys

    """

    def __init__(self, polysemous: Polysemous, context: str, target:str,
                 correct_definition_in_context: str, correct_definition_keys: List[str]):

        self._context: str = context
        self._target: str = target
        self._polysemous: Polysemous = polysemous
        self._correct_definitions_in_context: List[str] = correct_definition_in_context
        self._correct_definition_keys: List[str] = correct_definition_keys

    @property
    def definitions_list(self) -> List[str]:
        return self._polysemous.sense_definitions_list

    @property
    def polysemous(self) -> Polysemous:
        return self._polysemous

    @property
    def context(self) -> str:
        return self._context

    @property
    def target(self) -> str:
        return self._target

    @property
    def correct_definition_in_contexts(self):
        return self._correct_definitions_in_context

    @property
    def correct_definition_keys(self):
        return self._correct_definition_keys

    def to_json(self):
        json_dict = {}
        json_dict['context'] = self.context
        json_dict['target'] = self.target
        json_dict['polysemous'] = self.polysemous.to_json()
        json_dict['correct_definitions_in_context'] = self.correct_definition_in_contexts
        json_dict['correct_definition_key'] = self.correct_definition_keys
        return json_dict

    def __str__(self):
        re = 'name: {}; context: {}, sense key: {}'.format(self.polysemous.name, self.context, self.correct_definition_keys)
        return re
    pass


class MCQExperimentalData(ExperimentalData):
    """ Multiple Choice Experimental Data Entity Class
    This class extends the ExperimentalData class to include multiple-choice options for the polysemous word.
    Attributes:
        mode (Literal['1', 'A', 'a', '']): The format mode for displaying options. '1' for numeric, 'A' for uppercase letters, 'a' for lowercase letters, and '' for no prefix.
    Attributes:
        __options_list (MCQOptionsList): The list of multiple-choice options.
    Methods:
        options_str (str): Property to get the formatted string of options.
        correct_option_index (Set[int]): Property to get the set of indices of the correct options (0-based index).

    """
    mode: Literal['1', 'A', 'a', ''] = '1'
    def __init__(self, data: ExperimentalData, option_list: MCQOptionsList):
        super().__init__(data.polysemous, data.context, data.target,
                         data.correct_definition_in_contexts, data.correct_definition_keys)

        self.__options_list: MCQOptionsList = option_list

    @property
    def options_str(self) -> str:
        return self.__options_list.option_str(mode=MCQExperimentalData.mode)

    @property
    def correct_option_index(self) -> Set[int]:
        return self.__options_list.correct_options_index()
