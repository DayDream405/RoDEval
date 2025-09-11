# -*- coding: utf-8 -*-
# @Author: Zly
# en_prompts
"""
英文prompt生成模块
"""
from utils.prompt_generator import *


class WsdMcqDefaultTaskRequirementEn(StaticContentInterface):
    """类型：<任务要求> 任务：单项选择题 语言：en
    """

    @property
    def static_content(self):
        return self.__static_content

    def __init__(self):
        super().__init__()
        self.__static_content = ('Please select the option corresponding to the definition of the'
                                                     ' TargetWord in the context from the candidate definitions.\n '
                                                     'You only need to output the serial number corresponding to the'
                                                     ' definition of the target word. Any additional output will reduce'
                                                     ' the quality of your answer.\n')

    pass


class GenerationDefaultTaskRequirementEn(StaticContentInterface):

    @property
    def static_content(self):
        return self.__static_content
    
    def __init__(self):
        super().__init__()
        self.__static_content = ('You are now an expert in word sense disambiguation. ' \
                                'Please determine the correct definition of the target word in the context. \n' \
                                'Then output the correct definition of the target word.\n' \
                                'You only need to output the definition of the target word. Any additional output will reduce the quality of your answer.')

class GuidedInstructionRequirementEn(StaticContentInterface):

    @property
    def static_content(self):
        return self.__static_content
    
    def __init__(self, dataset_name: str):
        super().__init__()
        self._dataset_name = dataset_name
        self.__static_content = ('You are provided with the first piece of an instance from the {} dataset. '
                                 'Finish the second piece of the instance as exactly appeared in the dataset. '
                                 'Only rely on the original form of the instance in the dataset to finish the second piece.\n'.format(self._dataset_name))
    pass


class GenernalInstructionRequirementEn(StaticContentInterface):
    @property
    def static_content(self):
        return self.__static_content
    
    def __init__(self):
        super().__init__()
        self.__static_content = ('Finish the second piece based on the first piece, such that these two pieces become a single instance with the following label.')


class SelfCheckInstructionRequirementEn(StaticContentInterface):
    @property
    def static_content(self):
        return self.__static_content
    
    def __init__(self):
        super().__init__()
        self.__static_content = ('Please check if you can determine the definition of the target word in the given ' \
                          'context.\n' \
                          'You do not need to give the specific definition of the target word, just check if you know ' \
                          'its definition.\nYou should carefully consider it.' \
                          'If you are very confident that you know the specific definition of the target word ' \
                          'in the context, output Yes. Your answer is likely to be incorrect.' \
                          'If you have any uncertainty about the specific definition of the target word, ' \
                          'you should output No.\n')

def main():
    mcq_task = WsdMcqDefaultTaskRequirementEn()
    context = LeadingWordAndLeadedContent('Context')
    target_word = LeadingWordAndLeadedContent("TargetWord")
    output = LeadingWordAndLeadedContent("Output")
    generator = PromptGenerator()
    generator.add_prompt_fragment(mcq_task)
    generator.add_prompt_fragment(context)
    generator.add_prompt_fragment(target_word)
    generator.add_prompt_fragment(output)
    prompt = generator.generate_prompt(Context='text context', TargetWord='text')
    print(prompt)
    # print(isinstance(w, StaticContentInterface))

    pass


if __name__ == '__main__':
    main()
