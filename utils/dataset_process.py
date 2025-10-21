# -*- coding: utf-8 -*-
# @Author: Zly
# dataset_process
""" Processing public word sense disambiguation datasets (SemEval-07 -13 -15 and Senseval-2 -3)
        1. Read the xml file and parse the sentences and polysemous words in it
        2. Read the gold answer file and parse the keys of polysemous words
        3. Encapsulate the data into ExperimentalData objects
        4. Provide a unified dataset interface DataSetInterface
        5. Provide a unified dataset class UnifiedEvaluationDataSet
        6. Support many-to-many mapping of definitions and keys of polysemous words
    More datasets may be supported in the future
"""
from typing import List, Dict
from abc import ABC, abstractmethod
import xml.etree.ElementTree as Et

from entity.experimental_data import ExperimentalData, Polysemous
from utils.wordnet_tools import WordNetSynsets


class DatasetInterface(ABC):
    """ Unified Dataset Interface
    Attributes:
        data_list (List[ExperimentalData]): List of experimental data
        name (str): Name of the dataset
    """

    @property
    @abstractmethod
    def data_list(self) -> List[ExperimentalData]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
    pass


class UnifiedDataSet(DatasetInterface):
    """ Unified Dataset Class
    This class can process datasets in the format of SemEval-07 -13 -15 and Senseval-2 -3
    It reads the xml file and gold key file, and encapsulates the data into ExperimentalData objects
    It supports many-to-many mapping of definitions and keys of polysemous words

    Attributes:
        _dataset_name(str): dataset name
        _xml_file_path (str): xml file path
        _gold_key_file_path (str): gold key file path
        _key_book(Dict[str, str]): Dict[ID: Key]
        __data_list(List): List of ExperimentalData objects

    """
    def __init__(self, dataset_name: str, xml_file_path: str, gold_key_file_path: str):
        self._dataset_name = dataset_name
        self._xml_file_path = xml_file_path
        self._gold_key_file_path = gold_key_file_path
        self._key_book = self.__read_gold_key_file()
        self.__data_list = self.__read_xml()

    @property
    def data_list(self) -> List[ExperimentalData]:
        """

        Returns:
            List[ExperimentalData]: List of ExperimentalData objects

        """
        return self.__data_list

    @property
    def name(self) -> str:
        return self._dataset_name

    def __read_gold_key_file(self):
        """ Read the gold key file and parse it into a dictionary

        Returns:
            Dict[str,str]: Dict[ID: key]

        """
        keys = []
        id_key_dict = {}
        with open(self._gold_key_file_path) as f:
            for item in f.readlines():
                keys.append(item)

        for data in keys:
            id_key = data.split(' ')
            data_id = id_key[0]
            key = [k.replace('\n', '') for k in id_key[1:]]
            id_key_dict[data_id] = key
        return id_key_dict

    def __read_xml(self) -> List[ExperimentalData]:
        data_list: List[ExperimentalData] = []
        xml = Et.parse(self._xml_file_path)
        text_tags = xml.getroot().findall('text')
        for text_tag in text_tags:
            sentences = text_tag.findall('sentence')
            for s in sentences:
                new_data: List[ExperimentalData] = self.__parse_sentences(s)
                data_list.extend(new_data)

        return data_list

    def __parse_sentences(self, sentence_tag) -> List[ExperimentalData]:
        """ Parse a sentence node and encapsulate the data into ExperimentalData objects

        Args:
            sentence_tag: Sentence node

        Returns:
            List[ExperimentalData]: List of ExperimentalData objects

        """
        # 构建上下文
        # 定义不需要空格的标点符号和特殊字符
        PUNCTUATION = {
            '.', ',', ';', ':', '!', '?',      # 基本标点
            "'", '"', '`', '“', '”', '‘', '’', # 引号
            '(', ')', '[', ']', '{', '}',      # 括号
            '-', '–', '—',                     # 连字符和破折号
            '%', '#', '$', '&', '*',           # 特殊符号
            '/', '\\', '|',                    # 斜杠
            '+', '=', '<', '>',                # 数学符号
            '~', '^', '_',                     # 其他符号
            '@'                                # at符号
        }

        context = ''
        words = []
        for i, w in enumerate(sentence_tag):
            text = w.text
            # 处理空文本
            if not text:
                continue
                
            # 第一个单词直接添加
            if i == 0:
                words.append(text)
            else:
                # 检查当前词和前一个词
                prev_text = sentence_tag[i-1].text
                
                # 如果当前是标点符号或前一个是标点符号，不加空格
                if text in PUNCTUATION or prev_text in PUNCTUATION:
                    words.append(text)
                else:
                    words.append(' ' + text)
        context = ''.join(words)

        new_word_dict: Dict[str, Polysemous] = {}
        data_list: List[ExperimentalData] = []
        for w in sentence_tag:
            if w.tag =='instance':
                name = w.attrib['lemma']
                id = w.attrib['id']
                target = w.text
                new_word = None
                keys: List = self._key_book[id]
                for key in keys:
                    if key in new_word_dict:
                        new_word = new_word_dict[key]
                    else: # key not in new_word_list
                        new_word = Polysemous(name, WordNetSynsets.get_wn_synset(name))
                        new_word_dict[key] = new_word
                correct_definitions = [WordNetSynsets.get_wn_definition(k) for k in keys]
                data = ExperimentalData(new_word, context, target, correct_definitions, keys)
                data_list.append(data)

        return data_list

    pass

