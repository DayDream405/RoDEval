# -*- coding: utf-8 -*-
# @Author: Zly
# option_list
""" Multiple Choice Question Options List Entity Class

"""
from typing import Set, Literal


class MCQOptionsList(list):
    """ Multiple choice option list class, inherited from the list class, adds the function of recording correct options

        Attributes:
            __correct_option_book(Set[int]): Set of indices of correct options(index starts from 0)

        Methods:
            append(self, __object: str, is_correct=False): Add options. Is_comrrect=True indicates that the option is correct
            remove(self, __value): Remove option. Raises ValueError if the value is not present.
            option_str(self, mode: Literal['1', 'a', 'A', '']='1'): Get string representation of options list
            correct_options_index(self): Get the set of indices of correct options(index starts from 0)
        Inherits: list

    """
    def __init__(self):
        super().__init__()
        self.__correct_option_book: Set[int] = set()

    def append(self, __object: str, is_correct: bool=False) -> None:
        """Overloading list. append(). When adding the correct option, the index will also be added to the index collection.

        Args:
            __object(str): Added options
            is_correct(bool): Indicates whether the option is correct

        Returns:
            None

        """

        super().append(__object)
        if is_correct:
            self.__correct_option_book.add(super().__len__() - 1)
        pass

    def remove(self, __value: str):
        """Overloading list. remove(). When removing the correct option, the index will also be removed from the index collection.
           Raises ValueError if the value is not present.

        Args:
            __value: Removed option

        Returns:
            None

        """

        index = super().index(__value)
        if index in self.__correct_option_book:
            self.__correct_option_book.remove(index)
        super().remove(__value)
        pass

    def option_str(self, mode: Literal['1', 'a', 'A', '']='1') -> str:
        """Get string representation of options list

        Args:
            mode: Options representation mode. '1': numbered options; 'a': lowercase letter options; 'A': uppercase letter options; '': no prefix
                  If the number of options exceeds 26, it will automatically switch to '1' mode
        Returns:
            str: String representation of options list

        """

        return_str: str = ''
        m = mode
        if len(self) > 26:
            m = '1'
        if m == '1':
            i = 1
            for option in self:
                return_str += '{}.{} '.format(i, option)
                i += 1
        elif m == 'a':
            i = 97
            for option in self:
                return_str += '{}.{} '.format(chr(i), option)
                i += 1
        elif m == 'A':
            i = 65
            for option in self:
                return_str += '{}.{} '.format(chr(i), option)
                i += 1
        elif m == '':
            for option in self:
                return_str += '{} '.format(option)

        return return_str.rstrip()

    def correct_options_index(self) -> Set[int]:
        """Get the set of indices of correct options(index starts from 0)

        Returns:
            Set[int]: Set of indices of correct options(index starts from 0)

        """
        return self.__correct_option_book
    pass