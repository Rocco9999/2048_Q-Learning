from environment.Game2048_envMulti import Game2048_env
#from model.Dqn1_Model import DQNAgent
from model.Dqn2_ExperienceReplayModel import DQNAgent
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
from gymnasium.vector import AsyncVectorEnv

def log_debug_info(file_path, episode, actions, q_values, rewards, total_rewards, max_values):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        for i in range(len(actions)):
            writer.writerow([episode, actions[i], q_values[i], rewards[i], total_rewards[i], max_values[i]])

def make_env():
    return Game2048_env()

if __name__ == "__main__":
    # Numero di ambienti paralleli
    num_envs = 4
    envs = AsyncVectorEnv([make_env for _ in range(num_envs)])
    dummy_env = Game2048_env()  # Ambiente singolo per estrarre state_shape e action_space

    # Inizializza l'agente
    state_shape = dummy_env.observation_space.shape
    action_space = dummy_env.action_space.n
    agent = DQNAgent(state_shape, action_space)

    # File per salvare i log
    log_file = "debug_log.csv"
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Action", "Q-Values", "Reward", "Total-Reward", "Max Value"])

    # Variabili di controllo
    episodes = 0
    reached2048 = 0
    total_rewards = np.zeros(num_envs)

    # Loop principale
    states = envs.reset()
    while True:
        # Genera azioni per ogni ambiente
        actions = [agent.act(state) for state in states]
        
        # Esegui le azioni negli ambienti paralleli
        next_states, rewards, dones, infos = envs.step(actions)

        # Memorizza transizioni e aggiorna rewards
        for i in range(num_envs):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
            total_rewards[i] += rewards[i]
            
            if dones[i]:
                # Logging
                max_value = np.max(infos[i]['board'])  # Supponendo che il board sia in `infos`
                q_values = agent.model.predict(np.array(states[i])[np.newaxis, :])[0]
                log_debug_info(log_file, episodes + i, [actions[i]], [q_values.tolist()], [rewards[i]],
                               [total_rewards[i]], [max_value])
                
                # Salvataggio modello se necessario
                if max_value >= 2048:
                    reached2048 += 1
                    agent.save_model(f"dqn_model_2048_{episodes + i}.h5")
                    if reached2048 >= 10:
                        print(f"Raggiunto 2048 in 10 ambienti! Episodio {episodes + i}")
                        agent.save_model("dqn_model_2048_final.h5")
                        exit()

                # Resetta ambiente
                total_rewards[i] = 0

        # Addestra il modello con esperienze raccolte
        agent.replay()

        # Aggiorna gli stati
        states = next_states

        # Incrementa gli episodi completati
        episodes += num_envs

        # Log progressivo
        if episodes % 10 == 0:
            print(f"Episodi completati: {episodes}, Buffer: {len(agent.memory)}, Epsilon: {agent.epsilon}")