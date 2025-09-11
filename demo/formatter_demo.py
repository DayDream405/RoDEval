# -*- coding: utf-8 -*-
# @Author: Zly
# formatter_demo
""" 
    This module defines several formatter instances and their corresponding controllers,
    which can be used to format text in different ways, such as converting to lowercase,
    title case, removing punctuation, and converting to uppercase.
    Each formatter is encapsulated in a controller for easy management and application.
"""
from utils.formatter import *

lowercase_formatter = LowerCaseFormatter()
titlecase_formatter = TitleCaseFormatter()
remove_punctuation_formatter = RemovePunctuationFormatter()
uppercase_formatter = UpperCaseFormatter()

lowercase_controller = FormatterController()
lowercase_controller.add_formatter(lowercase_formatter)

titlecase_controller = FormatterController()
titlecase_controller.add_formatter(titlecase_formatter)

remove_punctuation_controller = FormatterController()
remove_punctuation_controller.add_formatter(remove_punctuation_formatter)

uppercase_controller = FormatterController()
uppercase_controller.add_formatter(uppercase_formatter)
