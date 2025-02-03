import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.Game2048_nopenalty_env import Game2048_env
#from model.Dqn1_Model import DQNAgent
from Project0.Dqn5_Efficient_Adaptive import DQNAgent
#from model.Dqn4_priorityBuffer import DQNAgent
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import os

"""Questo effettua l'addestramento una sola volta a fine gioco, quindi addestra su tutto il game e non sul singolo step, Inoltre utilizza
il Double Q Network, quindi viene integrato un modello target che viene utilizzato per calcolare i valori q del next state"""

def log_debug_info(file_path, episode, total_reward, max_tile, loss_history):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, total_reward, max_tile, loss_history[-1] if loss_history else "N/A"])

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
        plt.savefig('plots/resultModelNoPenaltyRegularize.png')

        plt.close()


def compress_state(state):
    """
    Converte una matrice di valori nel formato originale in una matrice compressa uint8.
    I valori sono rappresentati come esponenti di 2.
    """
    compressed = np.zeros_like(state, dtype=np.uint8)
    compressed[state > 0] = np.log2(state[state > 0]).astype(np.uint8)
    return compressed

def decompress_state(compressed):
    """
    Ripristina lo stato originale dalla rappresentazione compressa uint8.
    """
    decompressed = np.zeros_like(compressed, dtype=np.uint32)  # Usa uint32 per evitare overflow
    decompressed[compressed > 0] = 2 ** compressed[compressed > 0]
    return decompressed


if __name__ == "__main__":
    env = Game2048_env()
    state_shape = env.observation_space.shape  # Ottieni la forma dello stato
    action_space = env.action_space.n         # Numero di azioni
    agent = DQNAgent(state_shape, action_space)  # Inizializza l'agente DQN
    loss_history = []
    max_tile_list = []
    score_list = []


    # File per salvare i log
    log_file = "log_csv/debug_logModelNoPenaltyRegularize.csv"

    # Crea l'intestazione del file CSV
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward", "Max Value", "Loss"])

    counterPrint = 0
    episode = 0
    reached2048 = 0
    while True:
        state = env.reset()  # Inizializza l'ambiente
        done = False
        total_reward = 0
        episode_memory = []
        if os.path.exists("lastmodel.h5"):
            agent.load_model("lastmodel.h5")

        # print(env.game.board)
        # print("Ecco epsilon:", agent.epsilon)
        
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
            #print(f"Azione scelta: {action} quindi vado: {move}")

            # Esegue l'azione
            next_state, reward, done, info = env.step(action)

            
            compressed_state = compress_state(state)
            compressed_next_state = compress_state(next_state)

            episode_memory.append((compressed_state, action, reward, compressed_next_state, done))

            #print(env.game.board)

            # Aggiorna lo stato e accumula reward
            state = next_state
            total_reward += reward
            
        
        # Processa l'episodio alla fine
        agent.process_episode(episode_memory, env.score)

        if agent.loss_history:
            loss_history.append(agent.loss_history[-1])
        else:
            pass
        
        # Salva il modello se viene raggiunta una nuova soglia significativa
        max_tile = np.max(env.game.board)
        max_tile_list.append(max_tile)
        score_list.append(env.score)

        log_debug_info(log_file, episode, total_reward, max_tile, loss_history)

        print(f"Episode {episode}: Total Reward: {total_reward}, Max Tile: {max_tile}, Loss: {loss_history[-1] if loss_history else 'N/A'}, Epsilon: {agent.epsilon} , LR: {agent.current_lr} ")

        agent.save_model("lastmodel.h5")

        if max_tile >= 2048:
            reached2048 += 1
            agent.save_model(f"dqn_model_2048_{episode}.h5")
            if reached2048 >= 10:
                print(f"Ho raggiunto il 2048 all'episodio {episode}!")
                agent.save_model("dqn_model_2048.h5")
                break  # Esci dal ciclo infinito

        if max_tile >= 256 and episode > 100:
            agent.save_model(f"ModelNoPenaltyRegularize_{max_tile}_episode_{episode}.h5")
            print(f"Modello periodo salvato all'episodio: {episode} con numero max: {max_tile}")

        episode += 1  # Incrementa il numero di episodi
        

        
        plot_results(max_tile_list, score_list, loss_history)


    