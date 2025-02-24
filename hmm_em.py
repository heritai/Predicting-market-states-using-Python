# -*- coding: utf-8 -*-
"""EM_HMM.py

Implements an Expectation-Maximization (EM) algorithm for estimating the
parameters of a Hidden Markov Model (HMM) applied to financial time series data,
specifically designed to capture the dynamics of asset prices in a
high-frequency trading context.

This implementation is based on the methodology described in the paper:
"Modelling Asset Prices for Algorithmic and High-Frequency Trading"
(Álvaro Cartea & Sebastian Jaimungal).

The code includes:
    - Data simulation with hidden regimes.
    - HMM parameter estimation using EM algorithm.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

################################################################################
# 1. Simulation Functions
################################################################################

def simulFunc(params, n):
    """
    Simulates a time series with hidden Markov states.

    Args:
        params (dict): A dictionary containing the HMM parameters:
            'transition': Transition matrix (KxK, where K is the number of states).
            'lmbd': Rate parameters for exponential distributions (durations) for each state (K).
            'zprob': Probabilities of zero price revisions for each state (K).
            'sigma': Standard deviations for normal distributions (price revisions) for each state (K).
        n (int): The number of data points to simulate.

    Returns:
        tuple: A tuple containing:
            - latentZ (np.array): The sequence of hidden states.
            - log_mid_price (np.array): The simulated log-mid-price revisions.
            - time_between_trades (np.array): The simulated time durations between trades.
    """

    # Initialize the hidden state
    z = np.random.choice(len(params['lmbd']))
    # Sample the log price change, zprob is prob of zero event so 1-zprob is the prob of non-zero
    price = np.random.binomial(1, p=1 - params['zprob'][z]) * np.random.normal(scale=params['sigma'][z])
    # Sample time until the next trade
    tt = np.floor(np.random.exponential(scale=1 / params['lmbd'][z]))  # Censored data
    latentZ = [z]
    log_mid_price = [price]
    time_between_trades = [tt]

    for i in range(n - 1):
        # Sample the new state given the transitions
        z = np.random.choice(len(params['lmbd']), p=params['transition'][z])
        price = np.random.binomial(1, p=1 - params['zprob'][z]) * np.random.normal(scale=params['sigma'][z])
        tt = np.floor(np.random.exponential(scale=1 / params['lmbd'][z]))  # Censored

        latentZ.append(z)
        log_mid_price.append(price)
        time_between_trades.append(tt)

    return np.array(latentZ), np.array(log_mid_price), np.array(time_between_trades)


################################################################################
# 2.  Define HMM Parameters (Example for 2 Regimes)
################################################################################

# Example 2 regime simulation as in PCP
params2={'transition':[[.8,.2],[.43,.57]],
        'lmbd':[1.37,0.14],
        'zprob':[0.56,0.14],
        'sigma':[.00029,.00063]}

# Example 4 regime simulation as in AMZN

params4={'transition':np.array([[79.88,3.05,0.04,17.03],
                      [1.14,94.48,1.18,3.20],
                      [3.05,18.64,75.95,2.36],
                      [43.08,0.60,0.08,56.24]])/100,
        'lmbd':[2.614,2.101,1.203,0.487],
        'zprob':np.array([85.57,46.57,26.34,29.06])/100,
        'sigma':np.array([1.810,2.931,11.480,2.496])*1e-4}

def plot_simulation(time_process, price_process, latentZ, num_regimes):
    """Plots the simulated price path colored by hidden state.

    Args:
        time_process (np.array): Cumulative time between trades.
        price_process (np.array): Cumulative price.
        latentZ (np.array): Hidden state sequence.
        num_regimes (int): The number of hidden regimes.
    """
    marker_size = 40
    plt.figure(figsize=(9, 6))

    markers = ['^', 'o', '*', (4, 1)]  # Define markers for up to 4 regimes

    for i in range(num_regimes):
        plt.scatter(time_process[latentZ == i], price_process[latentZ == i],
                    alpha=0.7, s=marker_size, marker=markers[i % len(markers)]) #cycle through list if more than four regimes
    plt.xlabel("Time")
    plt.grid()
    plt.ylabel("Price")
    plt.legend([f'regime {i+1}' for i in range(num_regimes)])
    plt.title(f'Simulated Price Path with {num_regimes} Regimes')
    plt.show()

#Simulate and Plot 2 regimes
latentZ2, log_mid_price2, time_between_trades2 = simulFunc(params2, 2000)
price_process2 = 1 * np.exp(np.cumsum(log_mid_price2))
time_process2 = np.cumsum(time_between_trades2)
plot_simulation(time_process2, price_process2, latentZ2, 2)

#Simulate and plot 4 regimes
latentZ4, log_mid_price4, time_between_trades4 = simulFunc(params4, 5000)
price_process4 = 1 * np.exp(np.cumsum(log_mid_price4))
time_process4 = np.cumsum(time_between_trades4)
plot_simulation(time_process4, price_process4, latentZ4, 4)

################################################################################
# 3.  Likelihood, and EM algorithm
################################################################################

def log_mixture_density(t, x, theta):
    """
    Calculates the log-likelihood of observing a duration `t` and price revision `x`
    given the parameters `theta` for a single hidden state.  This is log(p(t,x|state)).
    """
    lmbd = theta['lmbd']
    zprob = theta['zprob']
    sigma = np.float64(theta['sigma'])

    log_cencored_time = -lmbd * t + np.log((1 - np.exp(-lmbd)))  # Log of censored exp dist
    log_log_price_revision = np.log(zprob) if x == 0 else np.log(1 - zprob) + norm.logpdf(x, scale=sigma) # Mixture component

    return log_cencored_time + log_log_price_revision


def alpha_forward(hmm_params, time, price):
    """
    Implements the forward algorithm to calculate alpha values.

    Args:
        hmm_params (dict): Dictionary containing HMM parameters (transition, lmbd, zprob, sigma, initProb).
        time (np.array): Array of time durations between trades.
        price (np.array): Array of price revisions.

    Returns:
        tuple: A tuple containing:
            - alpha_mat (np.array): The alpha matrix (probabilities of being in each state at each time).
            - scales (list): Scaling factors used to prevent underflow.
    """
    nb_states = len(hmm_params['lmbd'])
    # Each state has its own param values.
    theta = [{key: hmm_params[key][k] for key in ['lmbd', 'zprob', 'sigma']} for k in range(nb_states)]
    tr_mat = np.exp(hmm_params['log_transition'], dtype=np.longfloat)
    # the prior probabilites of starting in a given state at t=0 scaled by the liklihood of the first point
    alpha_old = [np.exp(p) * np.exp(log_mixture_density(time[0], price[0], theta[i])) for i, p in
                 enumerate(hmm_params['log_initProb'])]
    alpha_old = np.array(alpha_old, dtype=np.longfloat)
    alpha_old = alpha_old / np.sum(alpha_old)
    scales = [np.sum(alpha_old)]  # Initializing scales to prevent underflow
    alpha_list = [alpha_old]

    for t, x in zip(time[1:], price[1:]):
        ll = []
        for i in range(nb_states):
            s = 0
            for j in range(nb_states):
                s = s + alpha_old[j] * tr_mat[j][i] *  np.exp(log_mixture_density(t, x, theta[i])) # probability of the data
            ll.append(s)
        cc = np.sum(ll)
        ll = np.array(ll) / cc  # Scaling to prevent underflow
        scales.append(cc)
        alpha_list.append(ll)
        alpha_old = ll.copy()

    return np.array(alpha_list), scales

def beta_backward(hmm_params, scales, time, price):
    """
    Implements the backward algorithm to calculate beta values.

    Args:
        hmm_params (dict): Dictionary containing HMM parameters.
        scales (list): Scaling factors from the forward algorithm.
        time (np.array): Array of time durations.
        price (np.array): Array of price revisions.

    Returns:
        np.array: The beta matrix (probabilities of observing the remaining data given a state).
    """
    nb_states = len(hmm_params['lmbd'])
    nb_samples = len(time)
    theta = [{key: hmm_params[key][k] for key in ['lmbd', 'zprob', 'sigma']} for k in range(nb_states)] #each state it's on theta
    beta_old = np.full(nb_states, 1, dtype=np.longfloat)
    beta_list = [beta_old]
    tr_mat = np.exp(hmm_params['log_transition'], dtype=np.longfloat)

    for k in range(2, nb_samples + 1):
        ll = []
        for i in range(nb_states):
            s = 0
            for j in range(nb_states):
                s = s + beta_old[j] * np.exp(log_mixture_density(time[-k], price[-k], theta[j])) * tr_mat[i][j]
            ll.append(s)

        ll = np.array(ll) / scales[-k]  # Scaling to prevent underflow
        beta_list.append(ll)
        beta_old = ll.copy()

    return np.array(beta_list)


def log_epsilon_numerator(alphaMat, betaMat, time, price, hmm_params):
  """
  Calculates the unormalized Xi (Epsilon) values in log domain to prevent underflow.

  Xi(t,i,j) = P(Z_t = i, Z_{t+1} = j | data, model).

  Args:
      alphaMat (np.array): Alpha matrix from the forward algorithm.
      betaMat (np.array): Beta matrix from the backward algorithm.
      time (np.array): Array of time durations.
      price (np.array): Array of price revisions.
      hmm_params (dict): Dictionary containing HMM parameters.

  Returns:
      list: A list of lists containing the log of epsilon values.
  """
  log_tr_mat = hmm_params['log_transition']
  nb_states = len(hmm_params['lmbd'])
  nb_samples = len(time)
  theta = [{key: hmm_params[key][k] for key in ['lmbd', 'zprob', 'sigma']} for k in range(nb_states)]
  log_eps_list = []

  for t in range(nb_samples - 1):
    ilst = []
    for i in range(nb_states):
      jlst = []
      for j in range(nb_states):
        eps = (np.log(alphaMat[t][i])
               + log_tr_mat[i][j]
               + log_mixture_density(time[t + 1], price[t + 1], theta[j])
               + np.log(betaMat[-(t + 1)][j]))
        jlst.append(eps)
      ilst.append(jlst)
    log_eps_list.append(ilst)

  return log_eps_list

def log_liklihood(hmm_params, time, price, log_rtj, log_eps_numer):
    """
    Calculates the log-likelihood of the entire dataset given the model parameters.
    """
    nb_states = len(hmm_params['lmbd'])
    nb_samples = len(time)
    theta = [{key: hmm_params[key][k] for key in ['lmbd', 'zprob', 'sigma']} for k in range(nb_states)]

    term1 = 0
    for t in range(nb_samples):
        for j in range(nb_states):
            term1 = term1 + log_mixture_density(time[t], price[t], theta[j]) * np.exp(log_rtj[t][j])

    log_tr_mat = hmm_params['log_transition']
    term2 = 0
    for t in range(nb_samples - 1):
        for j in range(nb_states):
            for k in range(nb_states):
                term2 = term2 + log_tr_mat[j][k] * np.exp(log_eps_numer[t][j][k])

    log_pi = hmm_params['log_initProb']
    term3 = 0
    for j in range(nb_states):
        term3 = term3 + log_pi[j] * np.exp(log_rtj[0][j])

    return term1 + term2 + term3


def EM(initParams, time, price, nb_iter, frq=100, verbose=False):
  """
  Implements the Expectation-Maximization (EM) algorithm to estimate HMM parameters.

  Args:
      initParams (dict): Initial HMM parameters.
      time (np.array): Array of time durations.
      price (np.array): Array of price revisions.
      nb_iter (int): Number of EM iterations.
      frq (int): Frequency of printing intermediate results (for verbose mode).
      verbose (bool): If True, print progress information during EM iterations.

  Returns:
      tuple: A tuple containing:
          - logLike_list (list): List of log-likelihood values at each iteration.
          - hmm_params (dict): Estimated HMM parameters.
  """
  time = np.array(time)
  price = np.array(price)
  hmm_params = initParams.copy()
  updated_params = initParams.copy()
  logLike_list = []
  nb_states = len(hmm_params['lmbd'])
  nb_samples = len(time)
  loglike_old = 0

  for i in range(nb_iter):
    # E-step
    alpha_mat, scales = alpha_forward(hmm_params, time, price)
    beta_mat = beta_backward(hmm_params, scales, time, price)

    # this is the unormalized rtj in the log domain
    log_rtj = np.log(beta_mat[::-1] * alpha_mat)  # log of P(state|data), reverse beta for indexing
    # Epsilon is computed using numerator only
    log_eps_numer = log_epsilon_numerator(alpha_mat, beta_mat, time, price, hmm_params)
    # Caluclate log likelihood
    loglike = log_liklihood(hmm_params, time, price, log_rtj, log_eps_numer)
    # Check for Convergence
    if np.abs(loglike_old - loglike) < 1e-6:
      break
    loglike_old = loglike
    logLike_list.append(loglike)

    if verbose and i % frq == 0:
      print('iteration: ', i)
      print('logliklihood: ', loglike)
      param_copy = updated_params.copy()
      param_copy['log_transition'] = np.exp(param_copy['log_transition'])
      param_copy['log_initProb'] = np.exp(param_copy['log_initProb'])
      print(param_copy)

    # M-step: Parameter Update
    for j in range(nb_states):
      # The rate parameter is updated using a closed form solution that leverages the
      # distribution of the state to compute the mean duration
      updated_params['lmbd'][j] = -np.log(
          np.sum(np.exp(log_rtj[:, j]) * time) / np.sum(np.exp(log_rtj[:, j]) * (time + 1)))
      # The zero prob is updated using the number of time zero was hit (that is when
      # price==0) and the states values
      updated_params['zprob'][j] = np.sum(np.exp(log_rtj[price == 0, j])) / np.sum(np.exp(log_rtj[:, j]))
      # Calculate the state volatility based on the price revisions
      updated_params['sigma'][j] = np.sqrt(
          np.sum(np.exp(log_rtj[price != 0, j]) * (price[price != 0] ** 2)) / np.sum(np.exp(log_rtj[price != 0, j])))
      # Updating transtion probs must be done carefuly
      log_eps_numer_indexed = [t[j] for t in log_eps_numer]
      log_eps_matrix = np.array(log_eps_numer_indexed)
      new_transition = np.zeros(nb_states)
      for k in range(nb_states):
        epssjk = np.exp(log_eps_matrix[:,k])
        new_transition[k]=np.sum(epssjk)

      # Normalization of Transistion Probs with exponents
      new_transition = np.exp(new_transition)
      new_transition /= np.sum(new_transition)
      new_transition = np.log(new_transition)

      updated_params['log_transition'][j] = new_transition

      # update intial states
      updated_params['log_initProb'][j]=log_rtj[0,j]

    hmm_params = updated_params.copy()

  return logLike_list, hmm_params


################################################################################
# 4. Define Initial parameter values
################################################################################

#Init params for 2 regime
np.random.seed(1234) #For the 2 rgime market test
hmm_params1={'log_transition':np.log([[0.001,.999],[0.999,0.001]]),
            'lmbd':[1.47,.13],
            'zprob':[0.6,0.1],
            'log_initProb':np.log([.5,.5]),
            'sigma':[0.00021,0.00061]}

#Init params for 4 regime
np.random.seed(1532) # seed for the 4 regime code test
hmm_params4={'log_transition':np.log(np.array([[1.14,94.48,1.18,3.20],
                                              [79.88,3.05,0.04,17.03],
                                              [43.08,0.60,0.08,56.24],
                                              [3.05,18.64,75.95,2.36]
                                              ])/100),
        'lmbd':[2.614,2.101,1.203,0.487],
        'zprob':np.array([85.57,46.57,26.34,29.06])/100,
        'log_initProb':np.log([.25,.25,.25,.25]),
        'sigma':np.array([1.810,2.931,11.480,2.496])*1e-4}

################################################################################
# 5. EM execution on simulated data
################################################################################

# Simulate Data based on param2
latentZ2, log_mid_price2, time_between_trades2=simulFunc(params2,2000)
# Run for 400 iterations verbose every time
loglike2,params2=EM(hmm_params1,time_between_trades2,log_mid_price2,400,frq=1,verbose=True)

# Simulate Data based on param4
latentZ4, log_mid_price4, time_between_trades4=simulFunc(params4,5000)
# Run for 10 iterations verbose every time
loglike4,params4=EM(hmm_params4,time_between_trades4,log_mid_price4,10,frq=1,verbose=True)


################################################################################
# 6. Show Results
################################################################################

# Plot results from 2 regime simulation
plt.plot(loglike2)
plt.xlabel("iterations")
plt.ylabel("log liklihood")
plt.title("Log-Likelihood Evolution (2 Regimes)")
plt.show()

# Plot results from 4 regime simulation
plt.plot(loglike4)
plt.xlabel("iterations")
plt.ylabel("log liklihood")
plt.title("Log-Likelihood Evolution (4 Regimes)")
plt.show()