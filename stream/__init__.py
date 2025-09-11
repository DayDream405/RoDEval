from stream.file_stream import *
from stream.figure_stream import *
from entity.results import EvaluationResults

def main():
    path = ResultFilePath('deepseek-chat', 'classification-normal', 'semeval2007', 'option_recall')
    results: EvaluationResults = read_result_from_json(path)
    bar(results.get_result('option_recall'), 'option recall', 'id', 'recall')
    pass
if __name__ == '__main__':
    main()