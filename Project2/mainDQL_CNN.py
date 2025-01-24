import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.Game2048_nopenalty_env import Game2048_env
#from model.Dqn1_Model import DQNAgent
from Project2.Dqn7TestPERCNN import DQNAgent
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import os
#from memory_profiler import profile
#import tracemalloc


"""Questo è in fase di sviluppo, viene integrata la memoria a priorità e una diversa gestione di esplorazione"""

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
        plt.savefig('plots/resultCNN256.png')

        plt.close()

def control_matrix(matrix, matrix_modificata):

    righe = len(matrix)
    colonne = len(matrix[0]) if righe > 0 else 0

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
    #tracemalloc.start()


    # File per salvare i log
    log_file = "log_csv/debug_log_sequentialCNN256.csv"

    counterPrint = 0
    episode = 0
    save_directory, load_directory = "agent_savesCNN256", "agent_savesCNN256"
    resume = True
    start_episode = 700

    if resume is True:
        # Crea l'intestazione del file CSV
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Action", "Legal Moves", "Q-Values", "Reward", "Total Reward", "State", "Next State", "Done"])

        episode, max_tile_list, score_list, loss_history = agent.load_agent_state(load_directory, start_episode)
        resume = False
    else:
        # Crea l'intestazione del file CSV
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Action", "Legal Moves", "Q-Values", "Reward", "Total Reward", "State", "Next State", "Done"])



    while True:
        state = env.reset()  # Inizializza l'ambiente
        done = False
        total_reward = 0
        matrix_edit = np.zeros((4, 4), dtype=int)
        qxq_state = np.zeros((4, 4), dtype=int)
        
        while not done:
            # Sceglie un'azione
            state= env.game.board
            nice, qxq_state = control_matrix(state, matrix_edit)
            if nice:
                state = qxq_state
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

            processed_state = agent.state_to_binary_channels(state)
            q_values = agent.model.predict(processed_state.reshape(1,4,4,16), verbose=0)

            # Esegue l'azione
            next_state, reward, done, info = env.step(action)

            nice, qxq_state = control_matrix(next_state, matrix_edit)
            if nice:
                next_state = qxq_state

            # Salva la transizione
            agent.remember(np.array(state), action, reward, done)

            # Addestra il modello
            if episode > 2 and done:
                for _ in range(100):
                    agent.replay(episode)

            # Aggiorna lo stato e accumula reward
            
            total_reward += reward

            agent.update_epsilon()
            

            if done:
                print(env.game.board)


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
            #Salviamo lo stato attuale della matrice come il prossimo stato per partire
            env.game.board = next_state

        episode += 1  # Incrementa il numero di episodi

        if agent.loss_history:
            loss_history.append(agent.loss_history[-1])
        else:
            pass

        #Aggiornamento del modello target

        if (episode % 5) == 0:
            print("Aggiorno il modello target")
            agent.update_target_model()
            
        
        # Salva il modello se viene raggiunta una nuova soglia significativa
        max_tile = np.max(env.game.board)
        max_tile_list.append(max_tile)
        if max_tile > best_tile:
            best_tile = max_tile

        # if max_tile_list.count(512) > 20:
        #     agent.epsilon_min = 0.02
        #     if max_tile_list.count(1024) > 10:
        #         agent.epsilon_min = 0.01

        score_list.append(env.score)

        if max_tile >= 2048:
            agent.save_model(f"dqn_model_{max_tile}_2048_{episode}.h5")
            break  # Esci dal ciclo infinito
        elif max_tile >= 1024:
            agent.save_model(f"modelli1024/dqn_model_{max_tile}_2048_{episode}.h5")  
        elif max_tile >= 512:
            agent.save_model(f"modelli512/dqn_model_{max_tile}_2048_{episode}.h5")

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

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)


    