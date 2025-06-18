### Multi-Armed Bandit Algorithms

Different strategies for solving the exploration-exploitation dilemma in Multi-Armed Bandit (MAB) problems.

-   **Epsilon-Greedy:** A simple algorithm that balances exploration and exploitation using a fixed probability `epsilon`.
-   **UCB (Upper Confidence Bound):** An algorithm that selects arms based on their estimated value and an exploration bonus that accounts for uncertainty.
-   **Thompson Sampling:** A Bayesian algorithm that uses a probabilistic approach to balance exploration and exploitation.

**Files**
-   `epsilon_greedy_bernoulli.py`: Implements the Epsilon-Greedy algorithm for Bernoulli bandits.
-   `ucb_bernoulli.py`: Implements the UCB algorithm for Bernoulli bandits.
-   `thompson_sampling_bernoulli.py`: Implements the Thompson Sampling algorithm for Bernoulli bandits.