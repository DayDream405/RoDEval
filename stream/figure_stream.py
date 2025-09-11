# -*- coding: utf-8 -*-
# @Author: Zly
# figure stream
""" Contains a simple bar chart drawing function, and in the future, richer chart drawing functions will be added according to needs
"""
import matplotlib.pyplot as plt
from typing import Dict

def bar(data: Dict, save_path: str, title:str='Title', xlabel: str='X', ylabel: str='Y'):
    sorted_data = dict(sorted(data.items(), key=lambda item: int(item[0])))
    categories = list(sorted_data.keys())
    values = list(sorted_data.values())
    # Identify max and min values and their indices
    max_value = 1
    min_value = 0
    max_indices = [i for i, v in enumerate(values) if v == max_value]
    min_indices = [i for i, v in enumerate(values) if v == min_value]

    # Draw a bar chart
    plt.clf()
    bars = plt.bar(categories, values, color='skyblue', label='Normal Data')

    # Highlight max and min values
    # for idx in max_indices:
    #     plt.scatter(categories[idx], max_value, color='red', zorder=5)
    #     plt.text(categories[idx], max_value, 'Outlier', ha='center', va='bottom', fontsize=8, color='red')

    for idx in min_indices:
        plt.scatter(categories[idx], min_value, color='#dee2e6', zorder=7)
        # plt.text(categories[idx], min_value, 'Outlier', ha='center', va='top', fontsize=8, color='red')
    # Change the color of the bars for max and min values
    for idx in max_indices + min_indices:
        bars[idx].set_color('#dee2e6')  # Set color for outliers
    
    # Add legend
    # plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='#dee2e6', markersize=10)],
    #             loc='upper left', bbox_to_anchor=(0, 1.08))  # Adjust position as needed
    # Add titles and tags； fontsize=8
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Rotate the x-axis labels to avoid overlap
    plt.xticks(rotation=45, ha='right')
    # Display chart
    plt.tight_layout() # Adjust layout to prevent clipping of tick-labels
    plt.savefig(save_path, dpi=300)