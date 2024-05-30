import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of cells and genes
num_cells = 500
num_genes = 5

# Define parameters
p = np.random.uniform(0.1, 0.7, size=num_genes)  # Success probabilities for each category
n = np.random.uniform(2., 5., size=num_genes) # number of successes