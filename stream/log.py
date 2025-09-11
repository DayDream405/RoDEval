# -*- coding: utf-8 -*-
# @Author: Zly
# log
"""
Log stream, contains a logger loading function
"""
import logging


def load_logger(level=logging.INFO, log_file_path: str = None, name=__name__):
    """ Load a logger.

    Args:
        level (Literal, optional): log level. Defaults to logging.INFO.
        log_file_path (str, optional): log file path. Defaults to None.
        name (str, optional): Name of the object that generates the log. Defaults to __name__.

    Returns:
        Logger: logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d-%H:%M')
    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        logger.addHandler(file_handler)
        file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
