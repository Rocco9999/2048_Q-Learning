import gymnasium as gym
import numpy as np
from gymnasium import spaces
from math import log2
import collections
import math


##start
class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)  # Inizializza una griglia vuota
        self.add_number()
        self.add_number()

    def add_number(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = empty_cells[np.random.randint(0, len(empty_cells))]
            self.board[x, y] = 2 if np.random.random() < 0.9 else 4 #Con che probabilità vogliamo che venga spownato il numero 4?

    def move_left(self):
        moved = False
        score = 0
        for row in self.board:
            non_zero = row[row != 0]  # Rimuove gli zeri
            merged = []
            skip = False
            for i in range(len(non_zero)):
                #Questo skip serve per non fare una somma concatenata con valori già sommati
                if skip:
                    skip = False
                    continue
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged.append(non_zero[i] * 2)
                    score += non_zero[i] * 2
                    skip = True
                    moved = True
                else:
                    merged.append(non_zero[i])
            merged.extend([0] * (4 - len(merged)))  # Completa la riga con gli zero per avere dimensionalità 4
            if not np.array_equal(row, merged):
                moved = True
            row[:] = merged

        return moved, score

    def rotate_board(self):
        self.board = np.rot90(self.board) # ruota in senso antiorario

    def move(self, action):
        #Questa funzione ruota la tabella in maniera tale da fare sempre e soltanto operazioni di somma a sinistra
        #Semplifica la logica delle operazioni
        """0 = sinistra, 1 = sopra, 2 = destra, 3 = sotto"""
        moved = False
        for _ in range(action):
            self.rotate_board()
        moved, score = self.move_left()
        for _ in range(-action % 4):
            self.rotate_board()
        if moved:
            self.add_number()
        return moved, score

    def is_game_over(self):
        #Controllo per capire quando il gioco è effettivamente finito, non è possibile fare alcuna operazione 
        if np.any(self.board == 0):
            return False
        for action in range(4):
            test_board = self.board.copy()
            moved, _= self.move(action)
            if moved:
                self.board = test_board
                return False
        return True


class Game2048_env(gym.Env):
    rewards_buffer = collections.deque()
    iter = 0
    def __init__(self):
        super(Game2048_env, self).__init__()
        self.game = Game2048()
        self.score = 0
        self.move_score = 0
        self.penalty = 10
        self.previous_max = 2
        #Osservazioni per l'agente
        self.action_space = spaces.Discrete(4)  # Azioni: sinistra, sopra, destra, sotto
        self.observation_space = spaces.Box(0, 2048, shape=(4, 4), dtype=int)
        self.scaling_factor = 1.2  # Impostato così, ma è pissibile fare uno studio randomico per vedere come ottimizzarlo
        self.consecutive_action = None
        self.consecutive_count = 0
        self.max_consecutive_actions = 10
        self.last_consecutive_penalty = -1

    def step(self, action):
        valid, score = self.game.move(action)
        game_over = self.game.is_game_over() #Il gioco è terminato o con una vittoria o con una sconfitta
        max_number = np.max(self.game.board)
        reward = 0
        num_empty_cell = np.count_nonzero(self.game.board == 0) 
        self.move_score = score
        self.score += score
        done = False
         # Ora sarà reward a calcolare ovviamente penalità o bonus
        reward = self.calculate_reward(score=score, valid=valid, game_over=game_over, max_number=max_number)

        
        if action == self.consecutive_action:
            self.consecutive_count += 1
        else:
            self.consecutive_action = action
            self.consecutive_count = 1
            self.last_consecutive_penalty = -1

        if not valid and game_over:
            done = True
        
        # Se superiamo la soglia di azioni uguali consecutive
        if self.consecutive_count > self.max_consecutive_actions:
            if self.consecutive_count > 100:
                done = True
            penalty = max(self.last_consecutive_penalty * 1.1, -10)
            self.last_consecutive_penalty = penalty
            #print(f"Penalità per stallo: {penalty}")
            reward += penalty

        return self.game.board, reward, done, max_number
    
    def calculate_penalty(self, base_penalty, current_level):
        # Penalità dinamica decrescente ai livelli più alti
        return base_penalty / (1 + current_level)

    
    def calculate_reward(self, score, valid, game_over, max_number):
        reward = 0
        # ESPERIMENTO, RIFACIMENTO DELLA PROGRESSIONE NEL GIOCO
        # Divisione del gioco in livelli

        max_number = max(2, max_number)  # questo perchè ina lcuni casi ho visto che mi dava 0 come max

        # Calcoliamo il livello attuale (logaritmo in base 2)
        current_level = log2(max_number)

        # Bonus per aver superato un record precedente
        bonus_progress = 0
        if max_number > self.previous_max:
            bonus_progress = (current_level - log2(self.previous_max)) * (current_level ** self.scaling_factor)
            self.previous_max = max_number

        if not valid:
            # La mossa non sposta nulla
            if game_over:
                # Il gioco è finito
                if max_number in [512, 1024, 2048]:
                    # Reward per un livello massimo significativo
                    reward = bonus_progress + (current_level ** self.scaling_factor)
                else:
                    reward -= log2(max_number + 1)
            else:
                # Mossa non valida ma non è game over
                # Penalità proporzionale allo stato di avanzamento
                reward -= 0.1 * current_level
        else:
            # Caso: La mossa è valida
            # Al momento usiamo soltanto lo score come reward
            reward = score

            # Aggiungiamo l'eventuale bonus se abbiamo superato il precedente record
            if bonus_progress > 0:
                reward += bonus_progress
            elif bonus_progress == 0:
                reward += current_level * 0.05

            if max_number >= 512:
                reward += (current_level ** self.scaling_factor) * 2
                #Reward per quando supera il 512
        
        # Normalizziamo la reward
        normalized_reward = self.update_and_normalize(reward)
        #print(f"Reward grezza: {reward} reward normalizzata: {normalized_reward}")

        return normalized_reward


    def reset(self):
        #Resetta l'ambiente di gioco dopo aver terminato
        self.game = Game2048()
        self.score = 0
        return self.game.board

    def showMatrix(self):
        print(self.score)
        print(self.game.board)
    
    def update_and_normalize(self, reward):

        # Normalizzazione dinamica
        if reward >= 0:
            normalized_reward = min(log2(reward + 1), 10)
        else:
            normalized_reward = -min(log2(abs(reward - 1)), 10)
        
        return normalized_reward

if __name__ == "__main__":
    env = Game2048_env()
    env.showMatrix()

    done = False
    game_over = False
    numAction = 0
    while not done:
        # action = int(input("Scegli un'azione (0: sinistra, 1: sopra, 2: destra, 3: sotto): "))
        action = np.random.randint(0,3) #Per testing
        numAction += 1
        #print("L'azione scelta è stata: ", action)
        state, reward, done, max_number = env.step(action)
        #Ci salviamo anche lo stato in vista dell'interazione con l'agente
        env.showMatrix()
        print(f"Il numero più alto in griglia è: {max_number}")
        print(f"Reward: {reward}")

    #print("Numero di mosse: ", numAction)
    print("Game Over!")
