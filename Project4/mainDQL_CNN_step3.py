import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.Game2048_nopenalty_env import Game2048_env
#from model.Dqn1_Model import DQNAgent
from Project4.Dqn9NewCNNMedium import DQNAgent
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import os
#from memory_profiler import profile
#import tracemalloc


"""Questo funziona con il modello 9"""

def log_debug_info(file_path, episode, action, legal_move, reward, total_reward, state, next_state, done, memory_saved):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, episode, action, legal_move, reward, total_reward, state, next_state, done, memory_saved])

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
        plt.savefig('Project4/plots/resultCNN_NOPER_part2.png')

        plt.close()

def control_matrix(matrix):

    righe = len(matrix)
    colonne = len(matrix[0]) if righe > 0 else 0
    
    matrix_modificata = np.zeros((4, 4), dtype=int)

    if righe == 4 and colonne == 4:
        return False, matrix  # Già 4x4

    matrix_modificata = [riga[:] for riga in matrix]  # Crea una copia per evitare di modificare l'originale

    if righe > 4:
        # Controlla se sono presenti righe duplicate
        for i in range(righe):
            for j in range(i + 1, righe):
                if matrix_modificata[i] == matrix_modificata[j]:
                    del matrix_modificata[j]  # Elimina la riga duplicata
                    return (len(matrix_modificata) == 4 and len(matrix_modificata[0]) == 4,
                            matrix_modificata)

    if colonne > 4:
        # Controlla se sono presenti colonne duplicate
        for i in range(colonne):
            for j in range(i + 1, colonne):
                colonna_i = [riga[i] for riga in matrix_modificata]
                colonna_j = [riga[j] for riga in matrix_modificata]
                if colonna_i == colonna_j:
                    for riga in matrix_modificata:
                        del riga[j]  # Elimina la colonna duplicata
                    return (len(matrix_modificata) == 4 and len(matrix_modificata[0]) == 4,
                            matrix_modificata)

    return False, matrix_modificata


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
    best_tile = 0


    # File per salvare i log
    log_file = "Project4/log_csv/debug_log_sequentialCNN_NOPER_part2.csv"

    counterPrint = 0
    episode = 0
    save_directory, load_directory = "Project4/agent_savesCNN_NOPER", "Project4/agent_savesCNN_NOPER"
    resume = False
    start_episode = 0
    # model_path = os.path.join("agent_savesCNN_NOPER1", f"model_episode_{500}.h5")
    # agent.load_model(model_path)
    # agent.epsilon_decay = 0.9995

    if resume is True:
        # Crea l'intestazione del file CSV
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Action", "Legal Moves", "Reward", "Total Reward", "State", "Next State", "Done", "Ho salvato"])

        episode, max_tile_list, score_list, loss_history = agent.load_agent_state(load_directory, start_episode)
        resume = False
    else:
        # memory_path = os.path.join("agent_savesCNN256", f"model_episode_900.h5")
        # agent.load_model(memory_path)
        # Crea l'intestazione del file CSV
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Action", "Legal Moves", "Reward", "Total Reward", "State", "Next State", "Done", "Ho salvato"])



    while True:
        state = env.reset()  # Inizializza l'ambiente
        done = False
        total_reward = 0
        repeated_action_count = 0
        last_action = None
        memory_saved = None
        
        while not done:
            # Sceglie un'azione
            state= env.game.board
            nice, qxq_state = control_matrix(state)
            if nice:
                state = qxq_state

            #Blocco per limitare ristagnamenti
            legal_moves = []
            for action in range(4):
                # Check if the move is legal
                leagal, _= env.game.move(action, trial=True)
                if leagal:
                    legal_moves.append(action)

            action = agent.act(np.array(state))
            if action == last_action:
                repeated_action_count += 1
            else:
                repeated_action_count = 0

            if repeated_action_count > 20 and not memory_saved:
                # Forza un'esplorazione
                action = agent.act_ripetitive(state, legal_moves)

            last_action = action

            if(action == 0):
                move = "Sinistra"
            elif(action == 1):
                move = "Sopra"
            elif(action == 2):
                move = "Destra"
            elif(action == 3):
                move = "Giù"
            # print(f"Azione scelta: {action} quindi vado: {move}")

            # Esegue l'azione
            next_state, reward, done, info = env.step(action)

            nice, qxq_next_state = control_matrix(next_state)

            if nice:
                next_state = qxq_next_state

            # Salva la transizione
            memory_saved = agent.remember(np.array(state), action, reward, done, next_state)

            # Addestra il modello
            if done:
                print(env.game.board)
                for _ in range(100):
                    agent.replay(episode)

            # Aggiorna lo stato e accumula reward
            
            total_reward += reward

            log_debug_info(
                    log_file, 
                    episode, 
                    action, 
                    legal_moves, 
                    reward, 
                    total_reward, 
                    state, 
                    next_state, 
                    done,
                    memory_saved
                )
            #Salviamo lo stato attuale della matrice come il prossimo stato per partire
            env.game.board = next_state

        episode += 1  # Incrementa il numero di episodi

        if agent.loss_history:
            loss_history.append(agent.loss_history[-1])
        else:
            pass

        #Aggiornamento del modello target

        if (episode % 20) == 0:
            print("Aggiorno il modello target")
            agent.update_target_model()
            
        
        # Salva il modello se viene raggiunta una nuova soglia significativa
        max_tile = np.max(env.game.board)
        max_tile_list.append(max_tile)
        if max_tile > best_tile:
            best_tile = max_tile

        score_list.append(env.score)

        if max_tile >= 2048:
            agent.save_model(f"Project4/dqn_model_{max_tile}_2048_{episode}.h5")
            break  # Esci dal ciclo infinito
        elif max_tile >= 1024:
            agent.save_model(f"Project4/modelli1024/dqn_model_{max_tile}_2048_{episode}.h5")  
        elif max_tile >= 512:
            agent.save_model(f"Project4/modelli512/dqn_model_{max_tile}_2048_{episode}.h5")

        counterPrint += 1

        if (counterPrint == 1):
            print(f"Episode {episode}: Total Reward: {total_reward} Grandezza buffer: {agent.memory.nb_entries} Epsilon: {agent.epsilon}")
            #print(f"Episode: {episode}, State: {state}, Action: {action}, Q-Values: {q_values}, Reward: {reward}")
            counterPrint = 0

        

        
        if episode % 10 == 0:
            plot_results(max_tile_list, score_list, loss_history)

        if episode % 100 == 0 and episode != 0:
            arrays = {
                'max_tile_list' : max_tile_list,
                'score_list' : score_list,
                'loss_history' : loss_history
            }
            agent.save_agent_state(save_directory, episode, arrays)


    