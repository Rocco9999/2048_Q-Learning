import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
                    skip = True
                    moved = True
                else:
                    merged.append(non_zero[i])
            merged.extend([0] * (4 - len(merged)))  # Completa la riga con gli zero per avere dimensionalità 4
            if not np.array_equal(row, merged):
                moved = True
            row[:] = merged
        return moved

    def rotate_board(self):
        self.board = np.rot90(self.board) # ruota in senso antiorario

    def move(self, action):
        #Questa funzione ruota la tabella in maniera tale da fare sempre e soltanto operazioni di somma a sinistra
        #Semplifica la logica delle operazioni
        """0 = sinistra, 1 = sopra, 2 = destra, 3 = sotto"""
        moved = False
        for _ in range(action):
            self.rotate_board()
        moved = self.move_left()
        for _ in range(-action % 4):
            self.rotate_board()
        if moved:
            self.add_number()
        return moved

    def is_game_over(self):
        #Controllo per capire quando il gioco è effettivamente finito, non è possibile fare alcuna operazione 
        if np.any(self.board == 0):
            return False
        for action in range(4):
            test_board = self.board.copy()
            if self.move(action):
                self.board = test_board
                return False
        return True
    
    def calculateReward(self):
        zero_number = len(list(zip(*np.where(self.board == 0))))
        if zero_number > 0:
            print("Ci sono numeri vuoti")
        else:
            print("Non ci sono numeri vuoti")

class Game2048_env(gym.Env):
    def __init__(self):
        super(Game2048_env, self).__init__()
        self.game = Game2048()
        #Osservazioni per l'agente
        self.action_space = spaces.Discrete(4)  # Azioni: sinistra, sopra, destra, sotto
        self.observation_space = spaces.Box(0, 2048, shape=(4, 4), dtype=int)

    def step(self, action):
        prev_score = np.sum(self.game.board)
        valid = self.game.move(action)
        game_over = False
        max_number = 0
        if not valid:
            reward = -10 #Penalità nel caso in cui l'agente fa un'azione inconcludente
            done = True
            max_number = np.max(self.game.board)
            self.game.calculateReward()
        else:
            print("Score attuale:", np.sum(self.game.board))
            reward = np.sum(self.game.board) - prev_score
            done = False
            game_over = self.game.is_game_over() #Il gioco è terminato o con una vittoria o con una sconfitta
            max_number = np.max(self.game.board)
            self.game.calculateReward()
        return self.game.board, reward, done, game_over, max_number

    def reset(self):
        #Resetta l'ambiente di gioco dopo aver terminato
        self.game = Game2048()
        return self.game.board

    def showMatrix(self):
        print(self.game.board)

if __name__ == "__main__":
    env = Game2048_env()
    state = env.reset()
    env.showMatrix()

    done = False
    game_over = False
    while not game_over:
        # action = int(input("Scegli un'azione (0: sinistra, 1: sopra, 2: destra, 3: sotto): "))
        action = np.random.randint(0,3) #Per testing
        print("L'azione scelta è stata: ", action)
        state, reward, done, game_over, max_number = env.step(action)
        #Ci salviamo anche lo stato in vista dell'interazione con l'agente
        env.showMatrix()
        print(f"Il numero più alto in griglia è: {max_number}")
        print(f"Reward: {reward}")

    print("Game Over!")
