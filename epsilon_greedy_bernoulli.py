import numpy as np
import random

def epsilon_greedy_bernoulli(num_arms, true_probabilities, num_steps, epsilon):
    """
    Implements the epsilon-greedy algorithm for a Bernoulli multi-armed bandit problem.

    Args:
        num_arms (int): The number of arms in the bandit.
        true_probabilities (list): A list of the true success probability for each arm.
        num_steps (int): The total number of steps to run the algorithm.
        epsilon (float): The probability of exploration (choosing a random arm).

    Returns:
        float: The total reward obtained over the steps.
    """
    q_values = np.zeros(num_arms)  
    n_counts = np.zeros(num_arms)  
    total_reward = 0

    for _ in range(num_steps):
        
        if random.random() < epsilon:
            
            chosen_arm = random.randrange(num_arms)
        else:
            
            chosen_arm = np.argmax(q_values)
        
        reward = 1 if random.random() < true_probabilities[chosen_arm] else 0
        
        n_counts[chosen_arm] += 1
        q_values[chosen_arm] += (reward - q_values[chosen_arm]) / n_counts[chosen_arm]

        total_reward += reward

    return total_reward

if __name__ == '__main__':
    
    num_arms = 5
    true_probabilities = [0.3, 0.7, 0.5, 0.2, 0.6]
    num_steps = 1000
    num_trials = 100

    print(f"True success probabilities: {true_probabilities}")

    epsilon_values = [0.01, 0.1, 0.2]
    for eps in epsilon_values:
        average_total_reward = 0
        for _ in range(num_trials):
            average_total_reward += epsilon_greedy_bernoulli(num_arms, true_probabilities, num_steps, eps)
        average_total_reward /= num_trials
        print(f"Epsilon = {eps}: Average Total Reward over {num_trials} trials = {average_total_reward:.2f}")




