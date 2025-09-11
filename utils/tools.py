# -*- coding: utf-8 -*-
# @Author: Zly
# tools
"""
Some commonly used utility functions
"""
import re
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict


from sentence_transformers import SentenceTransformer, util

def remove_punctuation(s: str):
    """Remove punctuation marks from the incoming string

    Args:
        s: Input string

    Returns:
        str: String after removing punctuation marks

    """
    return re.sub(r'[^\w\s]', ' ', s)

def robust_remove_punctuation(text):
    """
    Reasonably remove punctuation marks and retain necessary formats (such as thousands of digits, hyphens, percent signs, etc.)
    Optimization: Avoid residual problems with PLAYHOLDER and enhance matching accuracy
    """
    # Reserved symbol patterns (stricter matching rules)
    preserved_patterns = [
        r'\([^)]+\)',            # Brackets and content (such as (see details))
        r'\d+-\d+',                # Number range (e.g. 70-77)
        r'\d{1,3}(?:,\d{3})+',  # Thousand separator (such as 1,000 or 100,000)
        r'\b\w+-\w+\b',          # Compound words with hyphens (such as state-of-the-art, but not matching isolated hyphens)
        r'-\w+-',                # Special markings (such as - LRB -, - RRB -)
        r'\d+\.\d+',             # Decimal (e.g. 3.14)
        r'\b[A-Za-z]+\.[A-Za-z]+\.?\b',  # Abbreviations (such as U.S., e.g.)
        r'\d+%',                 # Percentage sign (such as 50%)
    ]
    
    print(text)
    # 1. Protect the content that needs to be retained
    placeholders = {}
    for i, pattern in enumerate(preserved_patterns):
        for match in re.finditer(pattern, text):
            key = f"TEMP{i}{hash(match.group())}"  # Generate a unique placeholder
            key = key.replace('-', '')
            placeholders[key] = match.group()
            text = text.replace(match.group(), key)
    
    # 2. Remove punctuation marks
    punctuation = r'''!\"#$&'()*+,./:;<=>?@[\]^_`{|}~'''
    text = text.translate(str.maketrans(' ', ' ', punctuation))
    print(placeholders)
    print(text)
    # 3. Restore the protected content
    for key, value in placeholders.items():
        text = text.replace(key, value)
    
    # 4. Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def sentence_similarity(sentence1: str, sentence2: str, model_path: str=None):
    """Calculate sentence similarity

    Args:
        sentence1: sentence string 1
        sentence2: sentence string 2
        model_path: Custom embedding model path (optional); default is sentence-transformers/paraphrase-MiniLM-L6-v2

    Returns:
        sentence_transformers.util.pytorch_cos_sim(sentence1, sentence2)

    """
    path = 'sentence-transformers/paraphrase-MiniLM-L6-v2' # default model, you can change to other models
    if model_path is not None:
        path = model_path
    model = SentenceTransformer(path,
                               )
    encode1 = model.encode(sentence1)
    encode2 = model.encode(sentence2)
    similarity = util.pytorch_cos_sim(encode1, encode2).item()
    return similarity

embedding_llm = None
def sentence_similarity_llm(sentence1: str, sentence2: str, model_path: str=None):
    """Calculate sentence similarity using LLM embedding

    Args:
        sentence1: sentence string 1
        sentence2: sentence string 2
        model_path: Custom embedding model path (optional); default is Qwen/Qwen3-Embedding-4B

    Returns:
        model.similarity(encode1, encode2).item()

    """
    path = 'models/Qwen/Qwen3-Embedding-4B' # default model, you can change to other models
    if model_path is not None:
        path = model_path
    global embedding_llm
    if embedding_llm:
        model = embedding_llm
    else:
        model = SentenceTransformer(path,
                               )
        embedding_llm = model
    encode1 = model.encode(sentence1, prompt_name="query")
    encode2 = model.encode(sentence2)
    similarity = model.similarity(encode1, encode2).item()
    return similarity

def merge_dicts(dicts: List[Dict]) -> Dict:
    """
    Merge a list of dictionaries and calculate the average for each key.
    All dictionaries must have the same keys, and the values must be either numbers or dictionaries.
    Args:
        dicts (List[Dict]): List of dictionaries to be merged.
    """
    if not dicts:
        return {}

    # Check if the keys in all dictionaries are the same
    all_keys = set(dicts[0].keys())
    for d in dicts[1:]:
        if set(d.keys()) != all_keys:
            raise ValueError("All dictionaries must have the same keys")

    merged = {key: [] for key in all_keys}

    # Helper function to process values
    def process_value(value):
        if isinstance(value, dict):
            return merge_dicts([value])
        elif isinstance(value, (int, float)):
            return value
        else:
            raise ValueError("Values must be either numbers or dictionaries")

    # Collect values for each key
    for key in all_keys:
        for d in dicts:
            merged[key].append(process_value(d[key]))

    # Calculate averages
    averages = {}
    for key, values in merged.items():
        if isinstance(values[0], dict):
            averages[key] = merge_dicts(values)
        else:
            averages[key] = sum(values) / len(values)

    return averages

def log2rate(log: float) -> float:
    """ Convert log probability to percentage

    Args:
        log (float): log probability

    Returns:
        float: percentage
    """
    return np.round(np.exp(log)*100,2)

def pearson(x: List, y: List):
    """ Calculate Pearson correlation coefficient and p-value

    Args:
        x (List): A list of numerical values
        y (List): A list of numerical values

    Returns:
        correlation and p-value
    """
    correlation, p_value = pearsonr(x, y)
    return correlation, p_value

def calculate_variance(data):
    """ Calculate variance of a list of numbers"""
    return np.var(data, ddof=0)

def main():
    # Test the functions
    print('Some commonly used utility functions')
    pass

if __name__=="__main__":
    main()