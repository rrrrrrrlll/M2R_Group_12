import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from scipy.stats import nbinom, chisquare


def dfg(a, b):
    """
    return df of protein-generating gene of donor (a), CD(b)
    """
    file_path = 'datasets/Donor%d_CD%d_Genes.csv' %(a,b)
    all_df = pd.read_csv(file_path)
    file_path = 'datasets/mt_genes_metadata.csv'
    dfmeta = pd.read_csv(file_path)
    protein_coding_genes = dfmeta[dfmeta['gene_type'] == 'protein_coding']
    protein_names = protein_coding_genes['gene_name'].tolist()
    df2 = pd.DataFrame([all_df[i] for i in protein_names]).T
    return df2


def norm1(df, umi, const = None):
    if const is None:
        const = np.linalg.norm(umi)
    df1 = df.T
    print(np.linalg.norm(umi))
    umi1 = umi / const
    for i in range(len(umi1)):
        df1[i] = df1[i] / umi1[i]
    return df1.T

def norm2(df, s_n = 1000):
    df1 = df
    listfinal = [sum(df.iloc[i]) for i in range(len(df))]
    for i in range(len(df)):
        df1.iloc[i] = df1.iloc[i] / listfinal[i] * s_n
    return df1


def g_log(df_n, norm = norm2):
    df_final = norm(df_n).T
    gene_mean = [np.mean(df_final.iloc[i]) for i in range(13)]
    gene_var = [np.var(df_final.iloc[i]) for i in range(13)]
    log_gene_mean = np.log(gene_mean)
    log_gene_var = np.log(gene_var)
    return log_gene_mean, log_gene_var

def log_gene_plot(df_n, norm = norm2):
    log_gene_mean, log_gene_var = g_log(df_n, norm)
    plt.scatter(log_gene_mean,log_gene_var)


def meanvarplot(seq1):
    seq = seq1.T
    list_mean = [np.mean(seq.iloc[i]) for i in range(13)]
    list_var = [np.var(seq.iloc[i]) for i in range(13)]
    x = np.linspace(0,min(max(list_mean),max(list_var)),num = 1000)
    plt.plot(x,x)
    plt.scatter(list_mean, list_var)
    plt.show()
    print(list_var)
    print(list_mean)


def linfitplot(x,y):
    linear_model = pm.Model()
# Define the PyMC model
    with linear_model:
        # Priors for unknown parameters
        a_prior = pm.Normal('a', mu=0, sigma=10)
        b_prior = pm.Normal('b', mu=0, sigma=10)
        sigma_prior = pm.HalfNormal('sigma', sigma=2)

        # Expected value of outcome
        mu = a_prior * x + b_prior

        # Likelihood (sampling distribution) of observations
        vals = pm.Normal('vals', mu=mu, sigma=sigma_prior, observed=y)
        
        step = pm.Slice()
        tracel = pm.sample(1000, step=step, return_inferencedata=True)
        ppc_samples = pm.sample_posterior_predictive(tracel)
    
    ppc_vals = np.asarray(ppc_samples["posterior_predictive"].vals)
    
    for y_i in ppc_vals[0]:
        plt.plot(x,y_i,alpha=0.1,color="pink")
    
    plt.plot(x,y,"ko",label="Data")
    plt.legend()
    plt.xlabel("x")
    plt.xlabel("y")


def lse(a, b, norm = norm2):
    x = g_log(dfg(a,b), norm = norm)[0]
    y = g_log(dfg(a,b), norm = norm)[1]

    X = np.vstack([np.ones(len(x)), x]).T
    # Compute the least squares estimates using the normal equation
    # theta = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X.T @ X) @ X.T @ y

    # Extract the parameters
    a, b = theta
    print(f"Estimated parameters: a = {a}, b = {b}")

    # Predicted values using the fitted model
    y_pred = a + b * x

    # Calculate R-squared (R^2)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)

    print(f"R-squared (R^2) value: {r_squared}")

    # Plot the observed data and the fitted line
    plt.scatter(x, y, label='Observed data')
    plt.plot(x, y_pred, color='red', label='Fitted line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least Squares Estimation')
    plt.legend()
    plt.show()



def negative_binomial_gof(data, bins=None):
    """
    Perform a goodness-of-fit test to check if the data follows a negative binomial distribution.
    
    Parameters:
    - data: array-like, the observed data to test
    - bins: int or None, number of bins to group the data into for stabilization
    
    Returns:
    - chi_square_stat: float, the chi-square statistic
    - p_value: float, the p-value of the chi-square test
    - r: float, estimated parameter r (number of successes)
    - p: float, estimated parameter p (probability of success)
    """
    # Ensure data is a numpy array
    

    # Step 1: Calculate mean and variance of the data
    mean = np.mean(data)
    variance = np.var(data)
    
    # Step 2: Estimate the parameters r and p
    if variance <= mean:
        raise ValueError("Variance must be greater than the mean for a negative binomial distribution.")
    
    r = mean**2 / (variance - mean)
    p = mean / variance
    print(f"Estimated parameters: r = {r}, p = {p}")
    
    # Step 3: Create observed frequency table
    if bins is None:
        observed_freq = pd.Series(data).value_counts().sort_index()
    else:
        observed_freq, bin_edges = np.histogram(data, bins=bins)
        observed_freq = pd.Series(observed_freq, index=bin_edges[:-1])
    
    # Step 4: Calculate expected frequencies
    expected_freq = []
    for k in observed_freq.index:
        prob = nbinom.pmf(k, r, p)
        expected_freq.append(prob * len(data))
    
    # Convert to numpy arrays for the chi-square test
    observed_freq = observed_freq.values
    expected_freq = np.array(expected_freq)
    
    # Normalize expected frequencies to sum to observed frequencies
    expected_freq_sum = expected_freq.sum()
    observed_freq_sum = observed_freq.sum()
    expected_freq = expected_freq * (observed_freq_sum / expected_freq_sum)
    
    # Ensure no zero expected frequencies
    expected_freq = np.where(expected_freq == 0, 1e-10, expected_freq)
    
    print("Observed frequencies:\n", observed_freq)
    print("Expected frequencies:\n", expected_freq)
    
    # Step 5: Perform the chi-square goodness of fit test
    chi_square_stat, p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)
    
    # Output results
    print(f"Chi-Square Statistic: {chi_square_stat}")
    print(f"p-value: {p_value}")
    
    # Interpretation
    if p_value < 0.05:
        print("Reject the null hypothesis: The data does not follow a negative binomial distribution.")
    else:
        print("Fail to reject the null hypothesis: The data could follow a negative binomial distribution.")
    
    return chi_square_stat, p_value, r, p