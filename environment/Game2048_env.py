import gymnasium as gym
import numpy as np
from gymnasium import spaces
from math import log2
import collections
import math

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
    rewards_buffer = collections.deque(maxlen=100)
    def __init__(self):
        super(Game2048_env, self).__init__()
        self.game = Game2048()
        self.score = 0
        self.penalty = 10
        self.previous_max = 0
        #Osservazioni per l'agente
        self.action_space = spaces.Discrete(4)  # Azioni: sinistra, sopra, destra, sotto
        self.observation_space = spaces.Box(0, 2048, shape=(4, 4), dtype=int)

                # Esempio d'uso:
        self.normalizer = AdaptiveRewardNormalizer()

    def step(self, action):
        valid, score = self.game.move(action)
        game_over = self.game.is_game_over() #Il gioco è terminato o con una vittoria o con una sconfitta
        max_number = np.max(self.game.board)
        reward = 0
        num_empty_cell = np.count_nonzero(self.game.board == 0) 
        self.score += score
        done = False
         # Ora sarà reward a calcolare ovviamente penalità o bonus
        reward = self.calculate_reward(score=score, valid=valid, game_over=game_over, max_number=max_number)

        if not valid and game_over:
            done = True

        return self.game.board, reward, done, max_number
    
    def calculate_reward(self, score, valid, game_over, max_number):
        reward = 0
        # ESPERIMENTO, INTRODUZIONE DELLA PROGRESSIONE DI GIOCO
        # Aumentiamo la penalità per le mosse inutili proporzionalmente al max_number raggiunto.
        # Ad esempio: penalità = -1 all'inizio, e cresce all'aumentare del max_number o anche log2 di 32 sarà 5

        if max_number < 2:
            max_number = 2  # questo per evitare di avere zero come risultato perchè log2 di 1 è 0 
        penalty_scale = int(log2(max_number))

        # Bonus per aver superato un record precedente
        bonus_progress = 0
        if max_number > self.previous_max:
            # Più sarà grande il salto e più grande sarà il bonus, questo perchè stiamo effettivamente progredendo
            bonus_progress = max_number - self.previous_max
            self.previous_max = max_number
        
        if not valid:
            # La mossa non sposta nulla
            if game_over:
                # Il gioco è finito
                if max_number in [512,1024,2048]:
                    # Se la massima tessera raggiunta è tra queste,
                    # usiamo lo score come ricompensa
                    reward = bonus_progress + penalty_scale
                else:
                    ratio = max_number / 2048.0
                    penalty = -10 * (1 - ratio)
                    reward += penalty
            else:
                # Mossa non valida ma non è game over
                # Penalità proporzionale allo stato di avanzamento
                reward = -penalty_scale
        else:
            # Caso: La mossa è valida
            # Al momento usiamo soltanto lo score come reward
            reward = score

            # Aggiungiamo l'eventuale bonus se abbiamo superato il precedente record
            if bonus_progress > 0:
                reward += bonus_progress
        
        # Normalizziamo la reward
        normalized_reward, Game2048_env.rewards_buffer = self.normalizer.update_and_normalize(reward, Game2048_env.rewards_buffer)
        #print(f"Reward grezza: {reward}, Reward normalizzata: {normalized_reward}")

        return normalized_reward


    def reset(self):
        #Resetta l'ambiente di gioco dopo aver terminato
        self.game = Game2048()
        return self.game.board

    def showMatrix(self):
        print(self.score)
        print(self.game.board)

class AdaptiveRewardNormalizer:
    def __init__(self, k=3):
        self.k = k
        # Valori iniziali di fallback
        self.min_val = -10
        self.max_val = 2048
    
    def update_and_normalize(self, reward, rewards_buffer):
        # Aggiorna il buffer con la nuova reward
        rewards_buffer.append(reward)

        # Se non abbiamo abbastanza dati, usiamo i fallback (range statico)
        if len(rewards_buffer) < 10:
            normalize_static = self._normalize_static(reward)
            return normalize_static, rewards_buffer
        
        # Calcoliamo il min e max dal buffer
        current_min = min(rewards_buffer)
        current_max = max(rewards_buffer)

        # Normalizzazione asimmetrica dinamica:
        # Se reward < 0:
        #   map [current_min, 0] -> [-1,0]
        # Se reward >=0:
        #   map [0, current_max] -> [0,1]

        # Normalizziamo in base a questo range dinamico
        normalized_reward = self._dynamic_asymmetric_normalize(reward, current_min, current_max)
        
        return normalized_reward, rewards_buffer

    def _dynamic_asymmetric_normalize(self, reward, current_min, current_max):
        # Assicuriamoci che current_min <= 0 e current_max >=0
        # Se non è così, significa che non abbiamo reward negative o positive, gestiamo i casi estremi
        if current_min > 0:
            # Tutte le reward sono non negative
            # Normalizziamo come se [0, current_max] -> [0,1]
            # Se current_max == 0 evitiamo divisioni per zero
            if current_max == 0:
                return 0
            else:
                return min(reward/current_max,1)

        if current_max < 0:
            # Tutte le reward sono negative
            # Normalizziamo come se [current_min, 0] -> [-1,0]
            # Se current_min == 0 significa che reward=0, ma qui non può accadere dato current_max <0
            return -1 + (reward - current_min)/(-current_min)

        # Caso generale: abbiamo sia valori negativi che non negativi nel buffer
        if reward < 0:
            # Map [current_min,0] -> [-1,0]
            # formula: norm = -1 + (reward - current_min)*1/(0 - current_min)
            return -1 + (reward - current_min)/(-current_min)
        else:
            # reward >=0
            # Map [0, current_max] -> [0,1]
            if current_max == 0:
                return 0
            else:
                return min(reward/current_max,1)


    def _normalize_static(self, reward):
        # Anche nel fallback potremmo usare una versione semplificata della mappatura
        # Ad esempio, usiamo min_val e max_val come se fossero gli estremi iniziali
        # Negativi in [min_val,0], positivi in [0,max_val]
        if reward < 0:
            # [min_val,0] -> [-1,0]
            return -1 + (reward - self.min_val)/(-self.min_val)
        else:
            # [0,max_val] -> [0,1]
            if self.max_val == 0:
                return 0
            else:
                return min(reward/self.max_val,1)

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
        print("L'azione scelta è stata: ", action)
        state, reward, done, max_number = env.step(action)
        #Ci salviamo anche lo stato in vista dell'interazione con l'agente
        env.showMatrix()
        print(f"Il numero più alto in griglia è: {max_number}")

    print("Numero di mosse: ", numAction)
    print("Game Over!")
