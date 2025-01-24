import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.Game2048_nopenalty_env import Game2048_env
#from model.Dqn1_Model import DQNAgent
from ProjectOld.Dqn3 import DQNAgent
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

"""Questo è il DEEP Q LEARNING, funziona con l'ultimo ambiente sviluppato, il modello è il numero 3, non provare con altri modelli
perchè utilizziamo una memory di keras sviluppata appositamente per il RL che non necessita del next state come parametro"""

def log_debug_info(file_path, episode, action, legal_move, q_values, reward, total_reward, state, next_state, done):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, episode, action, legal_move, q_values, reward, total_reward, state, next_state, done])

def plot_results(max_tile, match_score, loss):
        # Plot Max Tile
        plt.figure(figsize=(12, 12))
        plt.subplot(311)
        plt.plot(range(1, len(max_tile) + 1), max_tile)
        plt.title('Max Tile per Game')
        plt.xlabel('Game')
        plt.ylabel('Max Tile')

        # Plot Match Score
        plt.subplot(312)
        plt.plot(range(1, len(match_score) + 1), match_score)
        plt.title('Score per Game')
        plt.xlabel('Game')
        plt.ylabel('Score')

        # Plot Loss
        plt.subplot(313)
        plt.plot(range(1, len(loss) + 1), loss)
        plt.title('Loss per Game')
        plt.xlabel('Game')
        plt.ylabel('Loss')

        plt.subplots_adjust(hspace=0.5)
        plt.savefig('plots/resultmain1.png')

        plt.close()


if __name__ == "__main__":
    # Verifica se TensorFlow rileva la GPU
    print("TensorFlow version:", tf.__version__)
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    env = Game2048_env()
    state_shape = env.observation_space.shape  # Ottieni la forma dello stato
    action_space = env.action_space.n         # Numero di azioni
    agent = DQNAgent(state_shape, action_space)  # Inizializza l'agente DQN
    loss_history = []
    max_tile_list = []
    score_list = []
    best_score = 0


    # File per salvare i log
    log_file = "log_csv/debug_log_main1.csv"

    # Crea l'intestazione del file CSV
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Action", "Legal Moves", "Q-Values", "Reward", "Total Reward", "State", "Next State", "Done"])

    counterPrint = 0
    episode = 0
    reached2048 = 0
    while True:
        state = env.reset()  # Inizializza l'ambiente
        done = False
        total_reward = 0
        # if os.path.exists("lastmodel.h5"):
        #     agent.load_model("lastmodel.h5")

        # print(env.game.board)
        # print("Ecco epsilon:", agent.epsilon)
        
        while not done:
            # Sceglie un'azione
            state= env.game.board
            legal_moves = []
            for action in range(4):
                # Check if the move is legal
                leagal, _= env.game.move(action, trial=True)
                if leagal:
                    legal_moves.append(action)

            action = agent.act(np.array(state), legal_moves)
            if(action == 0):
                move = "Sinistra"
            elif(action == 1):
                move = "Sopra"
            elif(action == 2):
                move = "Destra"
            elif(action == 3):
                move = "Giù"
            # print(f"Azione scelta: {action} quindi vado: {move}")

            q_values = agent.model.predict(np.expand_dims(state, axis=0), verbose=0)

            # Esegue l'azione
            next_state, reward, done, info = env.step(action)

            # Salva la transizione
            agent.remember(np.array(state), action, reward, done)

            # Addestra il modello
            agent.replay(episode)

            #print(env.game.board)

            # Aggiorna lo stato e accumula reward
            
            total_reward += reward
            

            if done:
                print(env.game.board)
                agent.update_epsilon()


            log_debug_info(
                    log_file, 
                    episode, 
                    action, 
                    legal_moves, 
                    q_values[0].tolist(),  # Salva i valori Q per ogni azione
                    reward, 
                    total_reward, 
                    state, 
                    next_state, 
                    done
                )
            # print("Stato inziziale")
            # print(state)
            # print("Next State")
            # print(next_state)
            env.game.board = next_state

        
        if agent.loss_history:
            loss_history.append(agent.loss_history[-1])
        else:
            pass
        
        # Salva il modello se viene raggiunta una nuova soglia significativa
        max_tile = np.max(env.game.board)
        max_tile_list.append(max_tile)
        score_list.append(env.score)
        if env.score >= best_score or (episode % 10) == 0:
            print("Salvo il modello")
            # agent.save_model("lastmodel.h5")
            best_score = env.score

        if max_tile >= 2048:
            reached2048 += 1
            agent.save_model(f"dqn_model_2048_{episode}.h5")
            if reached2048 >= 10:
                print(f"Ho raggiunto il 2048 all'episodio {episode}!")
                agent.save_model("dqn_model_2048.h5")
                break  # Esci dal ciclo infinito

        episode += 1  # Incrementa il numero di episodi
        
        if (counterPrint == 1):
            print(f"Episode {episode}: Total Reward: {total_reward} Grandezza buffer: {agent.memory.nb_entries} Epsilon: {agent.epsilon}")
            #print(f"Episode: {episode}, State: {state}, Action: {action}, Q-Values: {q_values}, Reward: {reward}")
            counterPrint = 0

        counterPrint += 1

        
        plot_results(max_tile_list, score_list, loss_history)


    