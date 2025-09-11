# -*- coding: utf-8 -*-
# @Author: Zly
# wordnet_tools
"""
    Obtain synonym sets and definitions through the wordnet interface of nltk
"""
from nltk.corpus import wordnet as wn

class WordNetSynsets:
    """Get synonym sets and definitions from WordNet."""
    wn_synsets = {}
    key_2_definition = {}

    @staticmethod
    def get_wn_synset(word_str: str) -> list:
        """Retrieve synonym sets through wordnet

        Args:
            word_str (str): a word string

        Returns:
            list: a list of synonym sets
        """
        syns = WordNetSynsets.wn_synsets.get(word_str, None)
        if syns is None:
            syns = []
            tag_word_synset = wn.synsets(word_str)
            for tag_word_syn in tag_word_synset:
                wn_word_definition = tag_word_syn.definition()
                syns.append(wn_word_definition)
            WordNetSynsets.wn_synsets[word_str] = syns
           
        return syns

    @staticmethod
    def get_wn_definition(key_str: str) -> str:
        """ Get definition through sense key

        Args:
            key_str (str): a sense key string

        Returns:
            str: the definition of the sense key
        """
        tag_definition = WordNetSynsets.key_2_definition.get(key_str, None)
        if tag_definition is None:
            tag_definition = wn.synset_from_sense_key(key_str).definition()
            WordNetSynsets.key_2_definition[key_str] = tag_definition
        
        return tag_definition