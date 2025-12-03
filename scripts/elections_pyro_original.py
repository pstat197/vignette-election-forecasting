# This file includes code adapted from Pyro tutorials
# Source: https://pyro.ai/examples/elections.html
# Licensed under the Apache License, Version 2.0

import pandas as pd
import torch
from pathlib import Path

# Path to the local data directory: project_root/data/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

electoral_college_votes = pd.read_pickle(DATA_DIR / "electoral_college_votes.pickle")
print(electoral_college_votes.head())
ec_votes_tensor = torch.tensor(
    electoral_college_votes.values, dtype=torch.float
).squeeze()


def election_winner(alpha):
    dem_win_state = (alpha > 0.0).float()
    dem_electoral_college_votes = ec_votes_tensor * dem_win_state
    w = (
        dem_electoral_college_votes.sum(-1) / ec_votes_tensor.sum(-1) > 0.5
    ).float()
    return w


frame = pd.read_pickle(DATA_DIR / "us_presidential_election_data_historical.pickle")
print(frame[[1976, 1980, 1984]].head())

results_2012 = torch.tensor(frame[2012].values, dtype=torch.float)
prior_mean = torch.log(results_2012[..., 0] / results_2012[..., 1])

idx = 2 * torch.arange(10)
as_tensor = torch.tensor(frame.values, dtype=torch.float)
logits = torch.log(as_tensor[..., idx] / as_tensor[..., idx + 1]).transpose(0, 1)
mean = logits.mean(0)
sample_covariance = (1 / (logits.shape[0] - 1)) * (
    (logits.unsqueeze(-1) - mean) * (logits.unsqueeze(-2) - mean)
).sum(0)
prior_covariance = sample_covariance + 0.01 * torch.eye(
    sample_covariance.shape[0]
)

import pyro
import pyro.distributions as dist


def model(polling_allocation):
    # This allows us to run many copies of the model in parallel
    with pyro.plate_stack("plate_stack", polling_allocation.shape[:-1]):
        # Begin by sampling alpha
        alpha = pyro.sample(
            "alpha",
            dist.MultivariateNormal(
                prior_mean, covariance_matrix=prior_covariance
            ),
        )

        # Sample y conditional on alpha
        poll_results = pyro.sample(
            "y",
            dist.Binomial(polling_allocation, logits=alpha).to_event(1),
        )

        # Now compute w according to the (approximate) electoral college formula
        dem_win = election_winner(alpha)
        pyro.sample("w", dist.Delta(dem_win))

        return poll_results, dem_win, alpha


std = prior_covariance.diag().sqrt()
ci = pd.DataFrame(
    {
        "State": frame.index,
        "Lower confidence limit": torch.sigmoid(prior_mean - 1.96 * std),
        "Upper confidence limit": torch.sigmoid(prior_mean + 1.96 * std),
    }
).set_index("State")
print(ci.head())

_, dem_wins, alpha_samples = model(torch.ones(100000, 51))
prior_w_prob = dem_wins.float().mean()
print("Prior probability of Dem win", prior_w_prob.item())

dem_prob = (alpha_samples > 0.0).float().mean(0)
marginal = torch.argsort((dem_prob - 0.5).abs()).numpy()
prior_prob_dem = pd.DataFrame(
    {
        "State": frame.index[marginal],
        "Democrat win probability": dem_prob.numpy()[marginal],
    }
).set_index("State")
print(prior_prob_dem.head())

import numpy as np


def correlation(cov):
    return cov / np.sqrt(
        np.expand_dims(np.diag(cov.values), 0)
        * np.expand_dims(np.diag(cov.values), 1)
    )


new_england_states = ["ME", "VT", "NH", "MA", "RI", "CT"]
cov_as_frame = pd.DataFrame(
    prior_covariance.numpy(), columns=frame.index
).set_index(frame.index)
ne_cov = cov_as_frame.loc[new_england_states, new_england_states]
ne_corr = correlation(ne_cov)
print(ne_corr)

southern_states = ["LA", "MS", "AL", "GA", "SC"]
southern_cov = cov_as_frame.loc[southern_states, southern_states]
southern_corr = correlation(southern_cov)
print(southern_corr)

cross_cov = cov_as_frame.loc[
    new_england_states + southern_states, new_england_states + southern_states
]
cross_corr = correlation(cross_cov)
print(cross_corr.loc[new_england_states, southern_states])

from torch import nn


class OutcomePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(51, 64)
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 1)

    def compute_dem_probability(self, y):
        z = nn.functional.relu(self.h1(y))
        z = nn.functional.relu(self.h2(z))
        return self.h3(z)

    def forward(self, y_dict, design, observation_labels, target_labels):
        pyro.module("posterior_guide", self)

        y = y_dict["y"]
        dem_prob = self.compute_dem_probability(y).squeeze()
        pyro.sample("w", dist.Bernoulli(logits=dem_prob))


prior_entropy = dist.Bernoulli(prior_w_prob).entropy()

from collections import OrderedDict

poll_in_florida = torch.zeros(51)
poll_in_florida[9] = 1000

poll_in_dc = torch.zeros(51)
poll_in_dc[8] = 1000

uniform_poll = (1000 // 51) * torch.ones(51)

# The swing score measures how close the state is to 50/50
swing_score = 1.0 / (
    0.5 - torch.tensor(prior_prob_dem.sort_values("State").values).squeeze()
).abs()
swing_poll = 1000 * swing_score / swing_score.sum()
swing_poll = swing_poll.round()

poll_strategies = OrderedDict(
    [
        ("Florida", poll_in_florida),
        ("DC", poll_in_dc),
        ("Uniform", uniform_poll),
        ("Swing", swing_poll),
    ]
)

from pyro.contrib.oed.eig import posterior_eig
from pyro.optim import Adam

eigs = {}
best_strategy, best_eig = None, 0

for strategy, allocation in poll_strategies.items():
    print(strategy, end=" ")
    guide = OutcomePredictor()
    pyro.clear_param_store()
    # To reduce noise when comparing designs, we will use the precomputed value of H(p(w))
    # By passing eig=False, we tell Pyro not to estimate the prior entropy on each run
    # The return value of `posterior_eig` is then -E_p(w,y)[log q(w|y)]
    ape = posterior_eig(
        model,
        allocation,
        "y",
        "w",
        10,
        12500,
        guide,
        Adam({"lr": 0.001}),
        eig=False,
        final_num_samples=10000,
    )
    eigs[strategy] = prior_entropy - ape
    print(eigs[strategy].item())
    if eigs[strategy] > best_eig:
        best_strategy, best_eig = strategy, eigs[strategy]

best_allocation = poll_strategies[best_strategy]
pyro.clear_param_store()
guide = OutcomePredictor()
posterior_eig(
    model,
    best_allocation,
    "y",
    "w",
    10,
    12500,
    guide,
    Adam({"lr": 0.001}),
    eig=False,
)

test_data = pd.read_pickle(DATA_DIR / "us_presidential_election_data_test.pickle")
results_2016 = torch.tensor(test_data.values, dtype=torch.float)
true_alpha = torch.log(results_2016[..., 0] / results_2016[..., 1])

conditioned_model = pyro.condition(model, data={"alpha": true_alpha})
y, _, _ = conditioned_model(best_allocation)

outcome = pd.DataFrame(
    {
        "State": frame.index,
        "Number of people polled": best_allocation,
        "Number who said they would vote Democrat": y,
    }
).set_index("State")
print(
    outcome.sort_values(
        ["Number of people polled", "State"], ascending=[False, True]
    ).head()
)

q_w = torch.sigmoid(guide.compute_dem_probability(y).squeeze().detach())
print("Prior probability of Democrat win", prior_w_prob.item())
print("Posterior probability of Democrat win", q_w.item())
