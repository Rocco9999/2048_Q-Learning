import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.Game2048_nopenalty_env import Game2048_env
#from model.Dqn1_Model import DQNAgent
from Project3.Dqn8TestNOPERCNN import DQNAgent
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
import os
from tqdm import tqdm
#from memory_profiler import profile
#import tracemalloc


"""Questo è stato addestrato due volte, la prima partendo da zero e la seconda caricando un modello pre addestrato
Le cartelle in cui salva e i file sono questi agent_savesCNN_NOPER_part2 e noper1. Ogni parte è il retrain, attualmente siamo al 3 retrain"""

def log_debug_info(file_path, episode, action, legal_move, reward, total_reward, state, done, memory_saved, game_step):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, episode, action, legal_move, reward, total_reward, state, done, memory_saved, game_step])

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
        plt.savefig('Project3/plots/resultCNN_NOPER_part6.png')

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
    prev_max_tile = 0
    prev_step = 0
    #tracemalloc.start()

    block_scores = []        # Lista con la media di max_tile di ogni blocco
    BLOCK_SIZE = 20          # Episodi per blocco

    # Queste 2 variabili servono per monitorare il blocco in corso
    current_block_max_tiles = []  # memorizza i max tile di questo blocco
    block_index = 0               # Contatore di blocchi conclusi
    restored_consecutive = 0


    # File per salvare i log
    log_file = "Project3/log_csv/debug_log_sequentialCNN_NOPER_part6.csv"

    counterPrint = 0
    episode = 0
    save_directory, load_directory = "Project3/agent_savesCNN_NOPER_part6", "Project3/agent_savesCNN_NOPER_part6"
    checkpoint_directory = "Project3/checkpoint"
    resume = True
    start_episode = 1900
    # model_path = os.path.join("Project3/agent_savesCNN_NOPER_part5", f"model_episode_{1000}.h5")
    # agent.load_model(model_path)
    # memory_path = os.path.join("Project3/agent_savesCNN_NOPER_part5", f"memory_episode_{1000}.pkl")
    # agent.load_memory(memory_path)
    # agent.epsilon_start = 0.01
    # agent.epsilon_decay = 0.9999

    if resume is True:
        # Crea l'intestazione del file CSV
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Action", "Legal Moves", "Reward", "Total Reward", "State", "Done", "Ho salvato", "Mosse"])

        episode, max_tile_list, score_list, loss_history = agent.load_agent_state(load_directory, start_episode)
        resume = False
    else:
        # memory_path = os.path.join("agent_savesCNN256", f"model_episode_900.h5")
        # agent.load_model(memory_path)
        # Crea l'intestazione del file CSV
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Action", "Legal Moves", "Reward", "Total Reward", "State", "Done", "Ho salvato", "Mosse"])



    while True:
        state = env.reset()  # Inizializza l'ambiente
        done = False
        total_reward = 0
        repeated_action_count = 0
        last_action = None
        memory_saved = None
        prev_step = agent.step_counter

        while not done:
            game_step = agent.step_counter - prev_step
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

            #if repeated_action_count > 5 and not memory_saved:
            if not memory_saved:
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

            if done:
                final_tile = np.max(next_state)
                semi_final_tile = np.sort(next_state.flatten())[-2] if next_state.size > 1 else 0
                if final_tile >= 2048:
                    reward += 100
                elif final_tile >= 1024:
                    if semi_final_tile >= 1024:
                        reward += 50
                    else:
                        reward += 0
                else:
                    pass

            nice, qxq_next_state = control_matrix(next_state)
            if nice:
                next_state = qxq_next_state

            # Salva la transizione
            memory_saved = agent.remember(np.array(state), action, reward, done, next_state)

            # Addestra il modello
            if done:
                print(env.game.board)
                for _ in tqdm(range(100), desc="Train progress"):
                    agent.replay(episode)

            #Abbassiamo il learning rate del 2% ogni volta che raggiunge il 2048
            current_lr = agent.change_lr_function()

            # Aggiorna lo stato e accumula reward
            total_reward += reward

            log_debug_info(log_file, episode, action, legal_moves, reward, total_reward, state, done, memory_saved, game_step)

            #Salviamo lo stato attuale della matrice come il prossimo stato per partire
            env.game.board = next_state

        episode += 1  # Incrementa il numero di episodi

        if agent.loss_history:
            loss_history.append(agent.loss_history[-1])
        else:
            pass
        
        # Salva il modello se viene raggiunta una nuova soglia significativa
        max_tile = np.max(env.game.board)
        max_tile_list.append(max_tile)
        if max_tile > best_tile:
            best_tile = max_tile

        score_list.append(env.score)

        if max_tile >= 2048:
            agent.save_model(f"Project3/dqn_model_{max_tile}_2048_{episode}.h5")
            #break  # Esci dal ciclo infinito
        elif max_tile >= 1024:
            agent.save_model(f"Project3/modelli1024/dqn_model_{max_tile}_2048_{episode}.h5")  
            print("Trovato il 1024")
        elif max_tile >= 512:
            print("Trovato il 512")
            #agent.save_model(f"Project3/modelli512/dqn_model_{max_tile}_2048_{episode}.h5")

        # INIZIO OPERAZIONI PERIODICHE 
        # OGNI 10 EPISODI AGGIORNO IL GRAFICO
        # OGNI 20 EPISODI AGGIORNO IL MODELLO TARGET E FACCIO CONTROLLI PER ROLLBACK
        # OGNI 50 EPISODI CANCELLO DAL BUFFER LE PEGGIORI 10 ESPERIENZE NEGATIVE
        # OGNI 100 EPISODI MI FACCIO UN SALVATAGGIO ROBUSTO DI TUTTO PER RIPRISTINARLO IN CASO DI ERRORI E STOP
        
        if episode % 10 == 0:
            plot_results(max_tile_list, score_list, loss_history)

        #Aggiornamento del modello target
        if (episode % 20) == 0:
            print("Aggiorno il modello target")
            agent.update_target_model()

        # #BLOCCO DI ROLLBACK
        # current_block_max_tiles.append(max_tile)

        # if episode % BLOCK_SIZE == 0:
        #     block_index += 1
        #     avg_max_tile = np.mean(current_block_max_tiles)
        #     block_scores.append(avg_max_tile)
        #     print(f"Blocco {block_index}: media max tile={avg_max_tile:.2f}")

        #     checkpoint_name = "block_checkpoint"

        #     # Confronto con il blocco precedente
        #     if len(block_scores) > 1:
        #         prev_score = block_scores[-2]
        #         if (prev_score - avg_max_tile) > 50.0 and restored_consecutive < 2:
        #             # -> ROLLBACK
        #             print(f"Peggioramento: blocco {block_index} ({avg_max_tile:.2f}) < blocco {block_index-1} ({prev_score:.2f}). ROLLBACK!")
        #             # Carica tutto
        #             episode, max_tile_list, score_list, loss_history = agent.load_agent_state_checkpoint(checkpoint_directory, checkpoint_name)
                    
        #             # Sovrascrivo block_scores[-1] con prev_score per dire "non aggiorno quest'ultimo"
        #             block_scores[-1] = prev_score
        #             restored_consecutive += 1

        #         else:
        #             # Salva checkpoint di routine
        #             arrays = {'max_tile_list': max_tile_list, 'score_list': score_list, 'loss_history': loss_history, 'episode': episode}
        #             agent.save_agent_state_checkpoint(checkpoint_directory, episode, arrays, checkpoint_name)
        #             # Salviamo anche in block_checkpoints
        #             print("Nessun peggioramento, continuo regolare e salvo il checkpoint.")
        #             restored_consecutive = 0
        #     else:
        #         arrays = {'max_tile_list': max_tile_list, 'score_list': score_list, 'loss_history': loss_history, 'episode': episode}
        #         agent.save_agent_state_checkpoint(checkpoint_directory, episode, arrays, checkpoint_name)

        #     # reset del blocco
        #     current_block_max_tiles = []


        #pulisco la memoria ogni 50 episodi
        if episode % 50 == 0 and episode != 0:
            if agent.memory.nb_entries > agent.batch_size:
                agent.memory.clean_low_score_episodes(n_to_remove=10)


        #Salvo tutto ogni 100 episodi
        if episode % 100 == 0 and episode != 0:
            arrays = {
                'max_tile_list' : max_tile_list,
                'score_list' : score_list,
                'loss_history' : loss_history
            }
            agent.save_agent_state(save_directory, episode, arrays)

        #STAMPA
        print(f"Episode {episode}: Total Reward: {total_reward} Grandezza buffer: {agent.memory.nb_entries} Epsilon: {agent.epsilon} Numero di step: {game_step} LR: {current_lr}")
        #print(f"Episode: {episode}, State: {state}, Action: {action}, Q-Values: {q_values}, Reward: {reward}")

    