import numpy as np
import math
import random

def ucb_bernoulli(num_arms, true_probabilities, num_steps, c):
    """
    Implements the Upper Confidence Bound (UCB) algorithm for a Bernoulli multi-armed bandit problem.

    Args:
        num_arms (int): The number of arms in the bandit.
        true_probabilities (list): A list of the true success probability for each arm.
        num_steps (int): The total number of steps to run the algorithm.
        c (float): The exploration parameter.

    Returns:
        float: The total reward obtained over the steps.
    """
    q_values = np.zeros(num_arms)  
    n_counts = np.zeros(num_arms)  
    total_reward = 0

    for step in range(1, num_steps + 1):
        
        ucb_values = np.zeros(num_arms)
        for arm in range(num_arms):
            
            if n_counts[arm] == 0:
                chosen_arm = arm
                break
            else:
                
                ucb_values[arm] = q_values[arm] + c * math.sqrt(math.log(step) / n_counts[arm])
        else:
            
            chosen_arm = np.argmax(ucb_values)

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

    c_values = [0.1, 0.5, 1.0, 2.0]
    for c_val in c_values:
        average_total_reward = 0
        for _ in range(num_trials):
            average_total_reward += ucb_bernoulli(num_arms, true_probabilities, num_steps, c_val)
        average_total_reward /= num_trials
        print(f"UCB with c = {c_val}: Average Total Reward over {num_trials} trials = {average_total_reward:.2f}")


