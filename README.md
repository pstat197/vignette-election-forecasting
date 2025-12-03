# vignette-election-forecasting

This project explores how Bayesian statistical methods can be applied to U.S. presidential election forecasting. Using historical state-level election data and poll-derived updates, the model demonstrates how priors, likelihoods, and Monte Carlo simulation combine to estimate Electoral College win probabilities.

## Contributors

- Andrew Guerra
- Satvik Talchuru
- Naira Younas
- Jeff Loomis
- Max Chang

## Vignette Abstract

This vignette presents a simplified Bayesian framework for forecasting U.S. presidential election outcomes, focusing on how historical voting patterns and current polling data can be combined to estimate uncertainty in the Electoral College. Using publicly available election datasets from Pyro covering the years 1976 through 2016, the model constructs a multivariate normal prior distribution over each state's underlying partisan lean. The prior mean is based on the 2012 election results, and the prior covariance reflects how states have historically shifted together across election cycles.

The vignette then updates this prior using either synthetic polling data generated from true 2016 vote shares or poll counts created from 2016 Democratic and Republican percentages. These poll results serve as the likelihood within an importance sampling procedure. Many possible election scenarios are drawn from the prior, weighted according to how well they align with the observed polling, and used to approximate the posterior distribution of outcomes.

The final outputs include the prior probability of achieving at least 270 Electoral College votes and the posterior probability after incorporating poll information. This vignette illustrates the core elements of Bayesian modeling, including prior construction, likelihood evaluation, and Monte Carlo based posterior approximation, in a clear and accessible election forecasting context.

## Repository Contents

### **data/**
- **data/electoral_college_votes.pickle**  
  State-by-state Electoral College vote allocations.

- **data/us_presidential_election_data_historical.pickle**  
  Historical Democratic and Republican vote totals (1976–2012). Used to construct the multivariate normal prior over state partisanship in both scripts.

- **data/us_presidential_election_data_test.pickle**  
  2016 presidential election results used to produce “true” state-level preferences for posterior evaluation and synthetic polling.

### **scripts/**
- **scripts/elections.py**  
  Implements a simplified Bayesian election model using importance sampling. This script constructs a multivariate normal prior from historical state-level election results and updates it with synthetic or real poll data to estimate prior and posterior Democratic win probabilities. It represents a streamlined version of the original Pyro-based model and is the primary script explained in detail in the vignette.

- **scripts/elections_pyro_original.py**  
  Contains the original Pyro-based version of the election forecasting model. This script uses Pyro's probabilistic programming tools, including model and guide structures and expected information gain utilities. It served as a foundation for the development of the simplified approach presented in `elections.py`.


## References

1. Pyro Team. *Predicting the outcome of a US presidential election using Bayesian optimal experimental design*  
   https://pyro.ai/examples/elections.html

### Related Work

2. FiveThirtyEight. *A User’s Guide to FiveThirtyEight’s 2016 General Election Forecast.*  
   https://fivethirtyeight.com/features/a-users-guide-to-fivethirtyeights-2016-general-election-forecast/  
   This article provides context for how probabilistic election forecasting is implemented in practice and offers background that helps motivate the Bayesian perspective used in this project.

