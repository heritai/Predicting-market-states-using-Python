
# EM-HMM for High-Frequency Financial Data

This repository contains a Python implementation of an Expectation-Maximization (EM) algorithm for estimating the parameters of a Hidden Markov Model (HMM) specifically designed for analyzing high-frequency financial time series data. The methodology implemented here is largely based on the paper:

*   **"Modelling Asset Prices for Algorithmic and High-Frequency Trading" by Álvaro Cartea & Sebastian Jaimungal.** https://www.tandfonline.com/doi/full/10.1080/1350486X.2013.771515

## Overview

The goal is to model the dynamics of asset prices in a high-frequency trading context, capturing different market conditions (regimes) and their impact on price movements and trading activity.  An HMM is used to represent these regimes, and the EM algorithm is used to learn the model parameters from data.

## Key Features

*   **HMM for Financial Data:** Models asset prices using hidden states (regimes), price revisions, and durations between trades, capturing market microstructure dynamics.
*   **Explicit Modeling of Zero Price Revisions:** Addresses the importance of explicitly modeling periods where trades occur without price changes.
*   **Data Simulation:** Provides functions for simulating financial time series data with hidden Markov states, allowing for testing and validation of the algorithm.
*   **EM Algorithm:** Implements the EM algorithm for estimating HMM parameters from data.
*   **Numerical Stability:**  Uses logarithms extensively to prevent underflow issues in probability calculations.
*   **Clear and Documented Code:**  Well-commented code with docstrings explaining the purpose, arguments, and return values of each function.

## Repository Contents

*   `EM_HMM.py`: Contains the Python code for data simulation, HMM parameter estimation using the EM algorithm, and plotting functions.
*   `README.md`: This file (describing the repository).
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.

## Requirements

*   Python 3.x
*   NumPy
*   SciPy (specifically, `scipy.stats.norm`)
*   Matplotlib

You can install these packages using `pip`:

```bash
pip install numpy scipy matplotlib
```

## Usage

1.  **Clone the Repository:**

    ```bash
    git clone [repository URL]
    cd [repository name]
    ```

2.  **Run the Code:**

    Execute the `EM_HMM.py` script:

    ```bash
    python EM_HMM.py
    ```

    This will:

    *   Simulate financial time series data using pre-defined or customized parameters.
    *   Estimate the HMM parameters using the EM algorithm.
    *   Plot the log-likelihood over iterations to assess convergence.
    *   Plot the data simulation and highlight the regime where a value was generated from

3.  **Customize Parameters:**

    Modify the `params` dictionaries (e.g., `params2`, `params4`) to experiment with different HMM configurations.  These dictionaries define:

    *   `transition`: The transition matrix between hidden states.
    *   `lmbd`: The rates of trade arrival.
    *   `zprob`: Probabilities of zero price revisions.
    *   `sigma`: Standard deviations of price revisions.
    *   `log_initProb`: This is log of the probabilities for the initial states

4.  **Analyze Results:**

    The code will print the estimated HMM parameters to the console and generate plots of the log-likelihood function.  Analyze these outputs to assess the model's fit and identify characteristics of the different market regimes.

## Code Explanation

The `EM_HMM.py` file contains the following key functions:

*   **`simulFunc(params, n)`:** Simulates a time series with hidden Markov states.
    *   Takes HMM parameters and the desired number of data points as input.
    *   Generates a sequence of hidden states, log-mid-price revisions, and time durations.

*   **`log_mixture_density(t, x, theta)`:** Calculates the log-likelihood of observing a duration `t` and price revision `x` given the parameters `theta` for a single hidden state. This allows us to capture the price revision and time duration with the transition of a regime

*   **`alpha_forward(hmm_params, time, price)`:** Implements the forward algorithm to calculate alpha values. Alpha represents the probability of starting the process in each state. This method prevent underflow by utilizing scales to make sure that they are well defined

*   **`beta_backward(hmm_params, scales, time, price)`:** Implements the backward algorithm to calculate beta values, which represents the probability of being in the end state.

*   **`log_epsilon_numerator(alphaMat, betaMat, time, price, hmm_params)`:** Calculates the log-likelihood of transitions of i to j given the rest of the parameters

*   **`EM(initParams, time, price, nb_iter, frq=100, verbose=False)`:**  Implements the Expectation-Maximization (EM) algorithm to estimate the HMM parameters:
    * E-step: Caluclates the Alpha, Beta, the Liklihood, the transition probability
    * M-step: Caluclates the lmbd, zero prob, volatiltiy
    * The transition Matrix
    * The Initial State

## Future Work

*   **Integration with Real Data:** Adapt the code to load and process real-world financial data.
*   **Model Selection:** Implement criteria for selecting the optimal number of hidden states (e.g., Bayesian Information Criterion (BIC)).
*   **Trading Strategy Development:**  Develop and test trading strategies based on the estimated HMM parameters, as suggested in the original paper.
*   **Extend the model** Add news and events as components of the HMM

## Contributions

Contributions to this repository are welcome!  Feel free to submit pull requests with bug fixes, improvements, or new features.
