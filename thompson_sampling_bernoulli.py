

import numpy as np
import random

def thompson_sampling_bernoulli(num_arms, true_probabilities, num_steps):
    """
    Implements the Thompson Sampling algorithm for a Bernoulli bandit problem.

    Args:
        num_arms (int): The number of arms in the bandit.
        true_probabilities (list): A list of the true success probability for each arm.
        num_steps (int): The total number of steps to run the algorithm.

    Returns:
        float: The total reward obtained over the steps.
    """
    
    alphas = np.ones(num_arms)  
    betas = np.ones(num_arms)   
    total_reward = 0

    for _ in range(num_steps):
        sampled_probabilities = [np.random.beta(alphas[arm], betas[arm]) for arm in range(num_arms)]
        
        chosen_arm = np.argmax(sampled_probabilities)

        reward = 1 if random.random() < true_probabilities[chosen_arm] else 0

        if reward == 1:
            alphas[chosen_arm] += 1
        else:
            betas[chosen_arm] += 1

        total_reward += reward

    return total_reward

if __name__ == '__main__':
    
    num_arms = 5
    true_probabilities = [0.3, 0.7, 0.5, 0.2, 0.6]
    num_steps = 1000
    num_trials = 100

    print(f"True success probabilities: {true_probabilities}")

    average_total_reward = 0
    for _ in range(num_trials):
        average_total_reward += thompson_sampling_bernoulli(num_arms, true_probabilities, num_steps)
    average_total_reward /= num_trials
    print(f"Thompson Sampling: Average Total Reward over {num_trials} trials = {average_total_reward:.2f}")

