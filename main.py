from environment.Game2048_env import Game2048_env

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.q_table = defaultdict(lambda: np.zeros(action_space))  # Tabella Q inizializzata a zero
        self.lr = learning_rate  # Tasso di apprendimento
        self.gamma = discount_factor  # Fattore di sconto
        self.epsilon = exploration_rate  # Tasso di esplorazione
        self.epsilon_decay = exploration_decay  # Decadimento esplorazione
        self.epsilon_min = exploration_min  # Valore minimo di esplorazione
        self.action_space = action_space  # Numero di azioni disponibili

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)  # Esplorazione: azione casuale
        else:
            return np.argmax(self.q_table[state])  # Sfruttamento: azione con il valore Q più alto

    def update_q_value(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + (self.gamma * self.q_table[next_state][best_next_action] * (1 - done))
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":
    env = Game2048_env()
    agent = QLearningAgent(action_space=env.action_space.n)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()  # Inizializza l'ambiente
        state = tuple(state.flatten())  # Appiattisce la griglia per usarla come chiave nella tabella Q
        done = False
        game_over = False
        total_reward = 0

        while not game_over:
            action = agent.choose_action(state)  # L'agente sceglie un'azione
            next_state, reward, done, game_over, info = env.step(action)  # Esegue l'azione nell'ambiente
            next_state = tuple(next_state.flatten())  # Appiattisce il nuovo stato
            agent.update_q_value(state, action, reward, next_state, done)  # Aggiorna il valore Q
            state = next_state  # Passa allo stato successivo
            total_reward += reward  # Accumula la ricompensa

        agent.decay_exploration()  # Riduce il tasso di esplorazione
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    
    
    max_tiles = []