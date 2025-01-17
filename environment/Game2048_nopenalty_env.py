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
        self.add_number(self.board)
        self.add_number(self.board)
        self.moved_board = np.zeros((4, 4), dtype=int)

    def add_number(self, board):
        empty_cells = list(zip(*np.where(board == 0)))
        if empty_cells:
            x, y = empty_cells[np.random.randint(0, len(empty_cells))]
            board[x, y] = 2 if np.random.random() < 0.9 else 4 #Con che probabilità vogliamo che venga spownato il numero 4?

    def move_left(self, trial):
        moved = False
        score = 0
        for row in self.moved_board:
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
            if not trial:
                row[:] = merged

        return moved, score

    def rotate_board(self):
        self.moved_board = np.rot90(self.moved_board) # ruota in senso antiorario

    def move(self, action, trial = False):
        #Questa funzione ruota la tabella in maniera tale da fare sempre e soltanto operazioni di somma a sinistra
        #Semplifica la logica delle operazioni
        """0 = sinistra, 1 = sopra, 2 = destra, 3 = sotto"""
        moved = False
        self.moved_board = self.board.copy()
        for _ in range(action):
            self.rotate_board()
        moved, score = self.move_left(trial)
        for _ in range(-action % 4):
            self.rotate_board()
        if moved and not trial:
            self.add_number(self.moved_board)
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
        self.max_number = 0

    def step(self, action):
        valid, score = self.game.move(action)
        game_over = self.game.is_game_over() #Il gioco è terminato o con una vittoria o con una sconfitta
        max_number = np.max(self.game.moved_board)
        reward = 0
        self.move_score = score
        self.score += score
        done = False
         # Ora sarà reward a calcolare ovviamente penalità o bonus
        reward = self.calculate_reward1(score=score, valid=valid, game_over=game_over, max_number=max_number)

        if game_over:
            done = True

        return self.game.moved_board, reward, done, max_number

    def calculate_reward(self, score, valid, game_over, max_number):
        reward = 0
        # 1) Bonus se hai superato il max_number visto finora
        if max_number > self.max_number:
            self.max_number = max_number
            reward += np.log2(max_number)

        # 2) Se la partita è finita, aggiungi un bonus/log finale
        if game_over:
            reward += np.log2(max_number)

        # 3) Se la mossa è valida, dai un contributo in base al punteggio ottenuto
        if valid:
            reward += score / 10.0 + (np.log2(max_number) * 0.1)
        else:
            reward = -1
        return reward
    
    def calculate_reward1(self, score, valid, game_over, max_number):
        reward = 0

        # Bonus per la fusione di tessere
        if score > 0:
            reward += score * 0.1  # Bonus proporzionale al punteggio ottenuto

        # Bonus per l'avvicinamento di tessere di alto valore (esempio semplice)
        reward += np.log2(max_number) * 0.5

        # Bonus per spazi liberi
        empty_cells = np.count_nonzero(self.game.board == 0)
        reward += empty_cells * 0.1

        # Penalità per mosse inutili
        if not valid:
            reward -= 2

        # Penalità per la dispersione (esempio semplice)
        non_zero_elements = self.game.board[self.game.board != 0]
        if non_zero_elements.size > 0:
            center_of_mass = np.mean(np.argwhere(self.game.board != 0), axis=0)
            distances = np.linalg.norm(np.argwhere(self.game.board != 0) - center_of_mass, axis=1)
            dispersion_penalty = np.mean(distances) * 0.05
            reward -= dispersion_penalty

        # Bonus/Penalità finale
        if game_over:
            reward += np.log2(max_number)

        return reward


    def reset(self):
        #Resetta l'ambiente di gioco dopo aver terminato
        self.game = Game2048()
        self.score = 0
        return self.game.board

    def showMatrix(self):
        print(self.score)
        print(self.game.board)

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
