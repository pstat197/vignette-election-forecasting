# This file includes code adapted from Pyro tutorials
# Source: https://pyro.ai/examples/elections.html
# Licensed under the Apache License, Version 2.0

# Simplified Bayesian election model using swing states only
# Outputs TWO posterior results:
# 1. Using synthetic poll generated from 2016 results
# 2. Using actual 2016 vote percentages scaled to poll size

import pandas as pd
import torch
import numpy as np
import pyro
import pyro.distributions as dist

BASE_URL = "https://raw.githubusercontent.com/pyro-ppl/datasets/master/us_elections/"

# ============================================================
# LOAD DATA
# ============================================================

electoral_college_votes = pd.read_pickle(BASE_URL + "electoral_college_votes.pickle")
ec_votes_tensor = torch.tensor(electoral_college_votes.values,
                               dtype=torch.float).squeeze()

frame = pd.read_pickle(BASE_URL + "us_presidential_election_data_historical.pickle")

# Historical swing states (2000–2020)
swing_states = ['FL','PA','MI','WI','OH','NC','AZ','GA','NV']
swing_indices = [frame.index.get_loc(st) for st in swing_states]

# ============================================================
# PRIOR FROM HISTORICAL DATA
# ============================================================

results_2012 = torch.tensor(frame[2012].values, dtype=torch.float)
prior_mean = torch.log(results_2012[..., 0] / results_2012[..., 1])

idx = 2 * torch.arange(10)
all_results = torch.tensor(frame.values, dtype=torch.float)
logits = torch.log(all_results[..., idx] / all_results[..., idx + 1]).transpose(0, 1)

mean = logits.mean(0)
sample_cov = (1/(logits.shape[0] - 1)) * (
    (logits.unsqueeze(-1) - mean) * (logits.unsqueeze(-2) - mean)
).sum(0)

prior_covariance = sample_cov + 0.01 * torch.eye(sample_cov.shape[0])
prior_dist = dist.MultivariateNormal(prior_mean, covariance_matrix=prior_covariance)

# ============================================================
# NATIONAL OUTCOME FUNCTION
# ============================================================

def election_winner(alpha_logits):
    dem_win_state = (alpha_logits > 0).float()
    dem_votes = ec_votes_tensor * dem_win_state
    return (dem_votes.sum() >= 270).float()

# ============================================================
# POSTERIOR INFERENCE VIA IMPORTANCE SAMPLING
# ============================================================

def posterior_win_prob_given_y(y_obs, allocation, num_alpha_samples=5000):
    """Approximate P(Dem win | observed poll y_obs)."""
    alpha_samples = prior_dist.sample((num_alpha_samples,))
    dem_win = torch.stack([election_winner(a) for a in alpha_samples])

    binom = dist.Binomial(total_count=allocation, logits=alpha_samples)
    log_lik = binom.log_prob(y_obs).sum(-1)

    maxlog = log_lik.max()
    weights = torch.exp(log_lik - maxlog)

    return ((weights * dem_win).sum() / weights.sum()).clamp(1e-6, 1 - 1e-6)

# ============================================================
# PRIOR DEM WIN PROBABILITY (PRINT THIS)
# ============================================================

print("\n==============================================")
print("Computing PRIOR distribution…")
print("==============================================")

alpha_prior_samples = prior_dist.sample((25000,))
prior_wins = torch.stack([election_winner(a) for a in alpha_prior_samples])
prior_prob = prior_wins.mean().item()

print(f"\nPrior probability of DEMOCRATIC win (national): {prior_prob:.4f}")
print("\n==============================================\n")

# ============================================================
# LOAD TRUE 2016 RESULTS
# ============================================================

test_data = pd.read_pickle(BASE_URL + "us_presidential_election_data_test.pickle")
results_2016 = torch.tensor(test_data.values, dtype=torch.float)
true_alpha_2016 = torch.log(results_2016[..., 0] / results_2016[..., 1])

# ============================================================
# POLL SIZE AND ALLOCATION
# ============================================================

TOTAL_POLL = 1500
allocation = torch.zeros(51)
per_state = TOTAL_POLL // len(swing_states)

for st in swing_states:
    allocation[frame.index.get_loc(st)] = per_state

# Remainder → Florida
allocation[frame.index.get_loc('FL')] += TOTAL_POLL - allocation.sum()

# ============================================================
# OPTION 1 — SYNTHETIC POLL GENERATED FROM MODEL
# ============================================================

print("\nGenerating SYNTHETIC poll results based on 2016 true preferences...\n")

y_synth = torch.zeros(51)

for st in swing_states:
    idx = frame.index.get_loc(st)
    total_polled = allocation[idx]

    p_dem = torch.sigmoid(true_alpha_2016[idx])
    y_synth[idx] = dist.Binomial(total_count=total_polled, probs=p_dem).sample()

# ============================================================
# OPTION 2 — ACTUAL 2016 PERCENTAGES AS POLL RESULTS
# ============================================================

print("\nGenerating poll using ACTUAL 2016 vote percentages...\n")

y_real = torch.zeros(51)

for st in swing_states:
    idx = frame.index.get_loc(st)
    total_polled = allocation[idx].item()

    dem_votes = results_2016[idx, 0]
    rep_votes = results_2016[idx, 1]
    total_votes = dem_votes + rep_votes

    p_dem = dem_votes / total_votes
    y_real[idx] = (p_dem * total_polled).round()

# ============================================================
# DISPLAY BOTH POLLS (with percentage column)
# ============================================================

print("\nSynthetic poll (from model):")
df_synth = pd.DataFrame({
    "State": swing_states,
    "Number polled": [allocation[frame.index.get_loc(s)].item() for s in swing_states],
    "Dem respondents": [y_synth[frame.index.get_loc(s)].item() for s in swing_states]
})
df_synth["Percent Dem"] = (df_synth["Dem respondents"] / df_synth["Number polled"]).round(3)
df_synth = df_synth.set_index("State")
print(df_synth)


print("\nPoll from ACTUAL 2016 percentages:")
df_real = pd.DataFrame({
    "State": swing_states,
    "Number polled": [allocation[frame.index.get_loc(s)].item() for s in swing_states],
    "Dem respondents": [y_real[frame.index.get_loc(s)].item() for s in swing_states]
})
df_real["Percent Dem"] = (df_real["Dem respondents"] / df_real["Number polled"]).round(3)
df_real = df_real.set_index("State")
print(df_real)

# ============================================================
# COMPUTE BOTH POSTERIORS
# ============================================================

posterior_synth = posterior_win_prob_given_y(y_synth, allocation)
posterior_real = posterior_win_prob_given_y(y_real, allocation)

print("\n==============================================")
print("FINAL POSTERIOR RESULTS (Swing States Only)")
print("----------------------------------------------")
print(f"Posterior (synthetic poll): {posterior_synth.item():.4f}")
print(f"Posterior (using actual 2016 percentages): {posterior_real.item():.4f}")
print("==============================================\n")
