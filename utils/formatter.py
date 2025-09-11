# -*- coding: utf-8 -*-
# @Author: Zly
# formatter
"""
Format Tool
"""
from utils.tools import *

from abc import ABC, abstractmethod
from typing import List


class Formatter(ABC):
    """Format abstract class
    """

    @abstractmethod
    def action(self, text: str):
        """Format action
        Args:
            text: input text
        Returns:
            formatted text
        """
        pass


class TitleCaseFormatter(Formatter):
    """ Capitalize the First Letter of Each Word
    """

    def action(self, sentence: str) -> str:
        return sentence.title()


class UpperCaseFormatter(Formatter):
    """ All Uppercase
    """

    def action(self, sentence: str) -> str:
        return sentence.upper()


class RemovePunctuationFormatter(Formatter):
    """ Replace Punctuations with robust rules
    """

    def action(self, sentence: str) -> str:
        return robust_remove_punctuation(sentence)


class LowerCaseFormatter(Formatter):
    """ All Lowercase
    """

    def action(self, sentence):
        return sentence.lower()


class FormatterController(object):
    """ Format controller class

        Attributes:
            __formatter_list(List[Formatter]): Format Rule List

    """
    def __init__(self):

        self.__formatter_list: List[Formatter] = []
        pass

    def add_formatter(self, formatter: Formatter):
        """ Add formatting rules

        Args:
            formatter: Formatter instance

        Returns:
            None

        """

        self.__formatter_list.append(formatter)
        pass

    def remove_formatter(self, formatter: Formatter):
        """ Remove formatting rules only when they exist

        Args:
            formatter: Formatter instance

        Returns:
            None

        """

        if formatter in self.__formatter_list:
            self.__formatter_list.remove(formatter)

    def control(self, text: str) -> str:
        """ Format the text according to the set formatting rules

        Args:
            text: input text

        Returns:
            str: formatted text

        """

        formatted_text = text
        for f in self.__formatter_list:
            formatted_text = f.action(formatted_text)
        return formatted_text
    pass
