from environment.Game2048_env import Game2048_env
from model.Dqn1_Model import DQNAgent
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv

def log_debug_info(file_path, episode, action, q_values, reward, total_reward, max_value):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, action, q_values, reward, total_reward, max_value])


if __name__ == "__main__":
    env = Game2048_env()
    state_shape = env.observation_space.shape  # Ottieni la forma dello stato
    action_space = env.action_space.n         # Numero di azioni
    agent = DQNAgent(state_shape, action_space)  # Inizializza l'agente DQN


    # File per salvare i log
    log_file = "debug_log.csv"

    # Crea l'intestazione del file CSV
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Action", "Q-Values", "Reward", "Total-Reward", "Max Value"])

    counterPrint = 0
    episode = 0
    while True:
        state = env.reset()  # Inizializza l'ambiente
        done = False
        total_reward = 0

        print(env.game.board)
        
        while not done:
            # Sceglie un'azione
            action = agent.act(np.array(state))
            if(action == 0):
                move = "Sinistra"
            elif(action == 1):
                move = "Sopra"
            elif(action == 2):
                move = "Destra"
            elif(action == 3):
                move = "Giù"
            print(f"Azione scelta: {action} quindi vado: {move}")

            # Esegue l'azione
            next_state, reward, done, info = env.step(action)

            # Salva la transizione
            agent.remember(np.array(state), action, reward, np.array(next_state), done)

            # Addestra il modello
            agent.replay()

            print(env.game.board)

            # Aggiorna lo stato e accumula reward
            state = next_state
            total_reward += reward
            
            if done:
                log_debug_info(log_file, episode, action, agent.model.predict(np.array(state)[np.newaxis, :])[0],
                               reward, total_reward, np.max(env.game.board))
        
        # Salva il modello se viene raggiunta una nuova soglia significativa
        max_tile = np.max(env.game.board)
        if max_tile >= 2048:
            print(f"Ho raggiunto il 2048 all'episodio {episode}!")
            agent.save_model("dqn_model_2048.h5")
            break  # Esci dal ciclo infinito

        if max_tile >= 512 and episode % 50 == 0:
            agent.save_model(f"dqn_model_{max_tile}_episode_{episode}.h5")
            print(f"Modello periodo salvato all'episodio: {episode} con numero max: {max_tile}")

        episode += 1  # Incrementa il numero di episodi
        
        if (counterPrint == 1):
            print(f"Episode {episode}: Total Reward: {total_reward} Grandezza buffer: {len(env.rewards_buffer)}")
            #print(f"Episode: {episode}, State: {state}, Action: {action}, Q-Values: {q_values}, Reward: {reward}")
            counterPrint = 0

        counterPrint += 1
    