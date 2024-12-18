from environment.Game2048_env import Game2048_env

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv

class QLearningAgent:
    def __init__(self, total_epochs, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_min=0.01):
        self.q_table = defaultdict(lambda: np.zeros(action_space))  # Tabella Q inizializzata a zero
        self.lr = learning_rate  
        self.gamma = discount_factor  # Fattore di sconto
        self.epsilon = exploration_rate  
        self.epsilon_min = exploration_min  
        self.action_space = action_space  
        self.total_epochs = total_epochs  
        self.epsilon_decay_linear = (exploration_rate - exploration_min) / (total_epochs * 0.75) 
        # Punti di transizione
        self.first_decay_limit = total_epochs * 0.30
        self.second_decay_limit = total_epochs * 0.60  
        self.third_decay_limit = total_epochs * 0.80  

        # Calcolo dei decadimenti
        self.slow_decay_1 = (exploration_rate - (exploration_min * 1.5)) / self.first_decay_limit
        self.fast_decay = ((exploration_rate - exploration_min) - (exploration_min * 1.5)) / (self.second_decay_limit - self.first_decay_limit)
        self.slow_decay_2 = (exploration_min * 1.1 - exploration_min) / (self.third_decay_limit - self.second_decay_limit)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)  # Esplorazione: azione casuale
        else:
            return np.argmax(self.q_table[state])  # Sfruttamento: azione con il valore Q più alto

    def update_q_value(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + (self.gamma * self.q_table[next_state][best_next_action] * (1 - done))
        self.q_table[state][action] += self.lr * (target - self.q_table[state][action])

    def decay_exploration(self, current_epoch):
        if current_epoch < self.first_decay_limit:
            # Decadimento lento (fase 1)
            self.epsilon = max(self.epsilon_min * 1.5, self.epsilon - self.slow_decay_1)
        elif current_epoch < self.second_decay_limit:
            # Decadimento rapido (fase 2)
            self.epsilon = max(self.epsilon_min * 1.1, self.epsilon - self.fast_decay)
        elif current_epoch < self.third_decay_limit:
            # Decadimento molto lento (fase 3)
            self.epsilon = max(self.epsilon_min, self.epsilon - self.slow_decay_2)
        else:
            # Fisso al minimo (fase 4)
            self.epsilon = self.epsilon_min

def log_debug_info(file_path, episode, action, q_values, reward, total_reward, max_value):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, action, q_values, reward, total_reward, max_value])


if __name__ == "__main__":
    env = Game2048_env()
    num_episodes = 50000
    agent = QLearningAgent(num_episodes, action_space=env.action_space.n)

# File per salvare i log
    log_file = "debug_log.csv"

    # Crea l'intestazione del file CSV
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Action", "Q-Values", "Reward", "Total-Reward", "Max Value"])

    counterPrint = 0
    max_number_in_train = 0
    for episode in range(num_episodes):
        state = env.reset()  # Inizializza l'ambiente
        state = tuple(map(tuple, state))  # Appiattisce la griglia per usarla come chiave nella tabella Q
        done = False
        total_reward = 0
        if max_number_in_train < np.max(env.game.board):
            max_number_in_train = np.max(env.game.board)

        if max_number_in_train < 1024 and episode >= num_episodes - 1:
            num_episodes += 1

        while not done:
            action = agent.choose_action(state)  # L'agente sceglie un'azione
            next_state, reward, done, info = env.step(action)  # Esegue l'azione nell'ambiente
            next_state = tuple(map(tuple, next_state))  # Appiattisce il nuovo stato
             # Debug: Stampa le informazioni nel terminale
            q_values = agent.q_table[state]
            max_value = np.max(next_state)

            agent.update_q_value(state, action, reward, next_state, done)  # Aggiorna il valore Q
            state = next_state  # Passa allo stato successivo
            total_reward += reward  # Accumula la ricompensa
            
            if done:
                # Salva le informazioni nel file CSV
                log_debug_info(log_file, episode, action, q_values, reward, total_reward, max_value)
        
        if episode % 10 == 0:
            print(f"Non aggiorno da: {env.iter} passaggi")
        agent.decay_exploration(episode)  # Riduce il tasso di esplorazione
        if (counterPrint == 1000):
            print(f"Episode {episode}: Total Reward: {total_reward} Grandezza buffer: {len(env.rewards_buffer)}")
            #print(f"Episode: {episode}, State: {state}, Action: {action}, Q-Values: {q_values}, Reward: {reward}")
            counterPrint = 0

        counterPrint += 1
    