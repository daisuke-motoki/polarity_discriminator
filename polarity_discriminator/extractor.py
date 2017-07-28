# -*- coding:utf-8 -*-
u'''
単語抽出モジュール
'''
from abc import abstractmethod


class WordExtractor(object):
    """
    """
    @abstractmethod
    def __call__(self, text):
        pass

    def extract_words(self, text):
        raise NotImplementedError('extract_words must be implemented')
