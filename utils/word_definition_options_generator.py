# -*- coding: utf-8 -*-
# @Author: Zly
# word_definition_options_generator
"""Word Definition Options List Generator
    Generate a list of candidate word meanings for specific rules, used for word sense disambiguation multiple-choice questions
    Optional configuration parameters:
        Randomness
        Number of options
        Correct option location range
        Average semantic similarity of options
        Number of correct options
"""
import random
import sys
from typing import TypedDict, List
from abc import ABC, abstractmethod
import copy

from entity import option_list
from entity.option_list import MCQOptionsList
from entity.experimental_data import ExperimentalData, Polysemous


class OptionsGenerationKwargs(TypedDict, total=False):
    """Option Generator Configuration Parameters
        is_random: bool - Whether to randomize the order of options
        option_quantity: int - Number of options to generate
        correct_option_index_range: tuple[int, int] - Range for the index of correct options [start, end)
        correct_option_quantity: int - Number of correct options to include
    """
    is_random: bool
    option_quantity: int
    correct_option_index_range: tuple[int, int]
    correct_option_quantity: int


class OptionGeneratorInterface(ABC):
    """Option Generator Interface

        Methods:
            generate_options(self, experimental_data: ExperimentalData): Generate a list of options

    """

    @abstractmethod
    def generate_options(self, experimental_data: ExperimentalData) -> MCQOptionsList:
        """Generate a list of options

        Args:
            experimental_data(ExperimentalData): Experimental data instance

        Returns:
            MCQOptionsList: List of options instance

        """
        pass


class OptionsGeneratorDecoratorInterface(OptionGeneratorInterface):
    """Options generator decorator interface
        Methods:
            generate_options(self, experimental_data: ExperimentalData): Generate a list of options

    """

    def __init__(self, option_generator: OptionGeneratorInterface):
        """

        Args:
            option_generator: The option generator to be decorated
        """
        super().__init__()
        self._option_generator = option_generator

    @abstractmethod
    def generate_options(self, experimental_data: ExperimentalData) -> MCQOptionsList:
        """Generate a list of options

                Args:
                    experimental_data(ExperimentalData): Experimental data instance

                Returns:
                    MCQOptionsList: List of options instance

                """
        pass


class OptionsGeneratorFactory(object):
    """ Options Generator Factory Class
        Configurable parameters:
            is_random: bool - Whether to randomize the order of options
            option_quantity: int - Number of options to generate
            correct_option_index_range: tuple[int, int] - Range for the index of correct options [start, end)
            correct_option_quantity: int - Number of correct options to include

    """

    def __init__(self, is_random: bool = False, option_quantity: int = None, correct_option_index_range: tuple[int, int] = None,
                correct_option_quantity: int = None):

        self._is_random: bool = is_random
        self._option_quantity: int = option_quantity
        self._correct_option_index_range = correct_option_index_range
        self._correct_option_quantity = correct_option_quantity
        self.__generator = DefaultOptionGenerator()

        self.setting()  # set generator

    def setting(self):
        """ Configure the option generator based on the specified parameters
        """
        if self._is_random:
            self.__generator = RandomOptionGenerator(self.__generator)
        if self._option_quantity is not None:
            self.__generator = QuantityLimitedOptionGenerator(self.__generator, option_quantity=self._option_quantity)
        if self._correct_option_index_range is not None:
            self.__generator = CorrectIndexLimitedOptionGenerator(self.__generator,
                                                                  self._correct_option_index_range[0], self._correct_option_index_range[1])
        if self._correct_option_quantity is not None:
            self.__generator = CorrectOptionQuantityLimitedOptionGenerator(self.__generator, self._correct_option_quantity)
        pass

    @property
    def get_options_generator_instance(self) -> OptionGeneratorInterface:
        return self.__generator


class DefaultOptionGenerator(OptionGeneratorInterface):
    """Default Options Generator
        Generates options directly from the provided definitions list in the experimental data

    """

    def generate_options(self, experimental_data: ExperimentalData) -> MCQOptionsList:
        mcq_option_list = MCQOptionsList()
        original_option_list: List[str] = experimental_data.definitions_list
        correct_definitions = experimental_data.correct_definition_in_contexts
        for item in original_option_list:
            if item in correct_definitions:
                mcq_option_list.append(item, is_correct=True)
            else:
                mcq_option_list.append(item)
        return mcq_option_list


class RandomOptionGenerator(OptionsGeneratorDecoratorInterface):
    """Decorator for option generator to randomize the order of options

    """
    def generate_options(self, experimental_data: ExperimentalData) -> MCQOptionsList:
        original_option_list = self._option_generator.generate_options(experimental_data)

        mcq_option_list = MCQOptionsList()
        list_len = len(original_option_list)
        random_list = [i for i in range(list_len)]
        correct_options = original_option_list.correct_options_index()
        for i in random_list:
            if i in correct_options:
                mcq_option_list.append(original_option_list[i], is_correct=True)
            else:
                mcq_option_list.append(original_option_list[i])

        return mcq_option_list


class QuantityLimitedOptionGenerator(OptionsGeneratorDecoratorInterface):
    """Decorator for option generator to limit the number of options generated

    """

    def __init__(self, option_generator: OptionGeneratorInterface, option_quantity: int):
        """

        Args:
            option_generator: The option generator to be decorated
            option_quantity: Number of options to generate, ≤ original option list length
        """
        super().__init__(option_generator)
        self._option_quantity = option_quantity

    def generate_options(self, experimental_data: ExperimentalData) -> MCQOptionsList:
        original_option_list = self._option_generator.generate_options(experimental_data)
        mcq_option_list = MCQOptionsList()
        list_len = len(original_option_list)

        self._option_quantity = list_len if list_len < self._option_quantity else self._option_quantity
        correct_options = copy.deepcopy(original_option_list.correct_options_index())
        book = [i for i in range(list_len)] # Mapping of new and old option indices

        for i in correct_options: # Remove correct option indices from the mapping
            book.remove(i)

        # Randomly generate new correct option indexes
        random_pool = [i for i in range(self._option_quantity)]
        correct_options_index = random.sample(random_pool, len(correct_options))

        for i in range(self._option_quantity): # Generate a new list of options
            if i in correct_options_index:
                original_correct_option_index = correct_options.pop()
                mcq_option_list.append(original_option_list[original_correct_option_index], is_correct=True)
            else:
                mcq_option_list.append(original_option_list[book[i]])


        return mcq_option_list

class CorrectIndexLimitedOptionGenerator(OptionsGeneratorDecoratorInterface):
    """Decorator for option generator to limit the index range of correct options in the generated list

    """

    def __init__(self, option_generator: OptionGeneratorInterface, start = 0, end = sys.maxsize): # [start, end)
        """

        Args:
            option_generator: The option generator to be decorated
            start: Correct option index range start
            end: Correct option index range end
        """
        super().__init__(option_generator)
        self._start = start
        self._end = end

    def generate_options(self, experimental_data: ExperimentalData) -> MCQOptionsList:
        original_option_list = self._option_generator.generate_options(experimental_data)

        option_len = len(original_option_list)
        mcq_option_list = MCQOptionsList()
        original_correct_option_index = copy.deepcopy(original_option_list.correct_options_index())
        correct_option_index = set()
        if option_len <= self._start:
            return original_option_list
        if option_len == len(original_correct_option_index):
            return original_option_list
        while len(original_correct_option_index) > self._end - self._start:
            original_correct_option_index.pop()
        for i in original_correct_option_index: # Ensure correct options are within the specified range
            if self._start <= i < self._end:
                correct_option_index.add(i)
            else:
                r = 0
                while True:
                    r = random.randint(self._start, self._end - 1)
                    if r not in correct_option_index:
                        correct_option_index.add(r)
                        break

        for i in range(option_len):
            if i in correct_option_index:
                correct_index = original_correct_option_index.pop()
                mcq_option_list.append(original_option_list[correct_index], is_correct=True)
            elif i in original_option_list.correct_options_index(): # i not in correct_option_index
                while True:
                    r = random.randint(0, option_len - 1)
                    if r not in original_option_list.correct_options_index():
                        break
                mcq_option_list.append(original_option_list[r])
            else: # i not in correct_option_index and i not in original_option_list.correct_options_index()
                mcq_option_list.append(original_option_list[i])

        return mcq_option_list


class CorrectOptionQuantityLimitedOptionGenerator(OptionsGeneratorDecoratorInterface):
    """Decorator for option generator to limit the number of correct options in the generated list

    """

    def __init__(self, option_generator: OptionGeneratorInterface, correct_option_quantity: int):
        """

        Args:
            option_generator: The option generator to be decorated
            correct_option_quantity: Number of correct options to include, ≤ original correct options count
        """
        super().__init__(option_generator)
        self._correct_option_quantity = correct_option_quantity

    def generate_options(self, experimental_data: ExperimentalData) -> MCQOptionsList:
        original_option_list = self._option_generator.generate_options(experimental_data)
        option_len = len(original_option_list)
        correct_option_quantity = self._correct_option_quantity if self._correct_option_quantity < option_len else option_len
        mcq_option_list = MCQOptionsList()
        correct_option_index = copy.deepcopy(original_option_list.correct_options_index())
        original_correct_option_quantity = len(correct_option_index)
        if option_len <= correct_option_quantity:
            mcq_option_list = copy.deepcopy(original_option_list)
        elif original_correct_option_quantity > correct_option_quantity:
            incorrect_option_index = 0 # Randomly selected incorrect option index
            while True:
                incorrect_option_index = random.randint(0, option_len - 1)
                if incorrect_option_index not in correct_option_index:
                    break

            quantity_count = 0
            for i in range(option_len):
                if i in correct_option_index:
                    if quantity_count < correct_option_quantity:
                        mcq_option_list.append(original_option_list[i], is_correct=True)
                        quantity_count += 1
                    else:
                        mcq_option_list.append(original_option_list[incorrect_option_index])
                else:
                    mcq_option_list.append(original_option_list[i])
        elif original_correct_option_quantity < correct_option_quantity:
            new_correct_option_index = list(random.choices(list(correct_option_index), k=correct_option_quantity - original_correct_option_quantity))
            replaced_option_index = set()
            for _ in range(len(new_correct_option_index)):
                while True:
                    i = random.randint(0, option_len - 1)
                    if i not in replaced_option_index and i not in correct_option_index:
                        replaced_option_index.add(i)
                        break

            for i in range(option_len):
                if i in replaced_option_index:
                    mcq_option_list.append(original_option_list[new_correct_option_index.pop()], is_correct=True)
                elif i in correct_option_index:
                    mcq_option_list.append(original_option_list[i], is_correct=True)
                else:
                    mcq_option_list.append(original_option_list[i])
        else: # original_correct_option_quantity == self._correct_option_quantity
            mcq_option_list = copy.deepcopy(original_option_list)

        return mcq_option_list