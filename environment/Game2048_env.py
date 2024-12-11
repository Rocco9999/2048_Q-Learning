import gymnasium as gym
import numpy as np
from gymnasium import spaces

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)  # Inizializza una griglia vuota
        self.add_tile()
        self.add_tile()

    def add_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = empty_cells[np.random.randint(0, len(empty_cells))]
            self.board[x, y] = 2 if np.random.random() < 0.9 else 4

    def move_left(self):
        moved = False
        for row in self.board:
            non_zero = row[row != 0]  # Rimuovi gli zeri
            merged = []
            skip = False
            for i in range(len(non_zero)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                    merged.append(non_zero[i] * 2)
                    skip = True
                    moved = True
                else:
                    merged.append(non_zero[i])
            merged.extend([0] * (4 - len(merged)))  # Completa con zeri
            if not np.array_equal(row, merged):
                moved = True
            row[:] = merged
        return moved

    def rotate_board(self):
        self.board = np.rot90(self.board)

    def move(self, action):
        """Esegue l'azione specificata: 0 = sinistra, 1 = sopra, 2 = destra, 3 = sotto"""
        moved = False
        for _ in range(action):
            self.rotate_board()
        moved = self.move_left()
        for _ in range(-action % 4):
            self.rotate_board()
        if moved:
            self.add_tile()
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for action in range(4):
            test_board = self.board.copy()
            if self.move(action):
                self.board = test_board
                return False
        return True

class Game2048_env(gym.Env):
    def __init__(self):
        super(Game2048_env, self).__init__()
        self.game = Game2048()
        self.action_space = spaces.Discrete(4)  # Azioni: sinistra, sopra, destra, sotto
        self.observation_space = spaces.Box(0, 2048, shape=(4, 4), dtype=int)

    def step(self, action):
        prev_score = np.sum(self.game.board)
        valid = self.game.move(action)
        game_over = False
        if not valid:
            reward = -10
            done = True
        else:
            reward = np.sum(self.game.board) - prev_score
            done = False
            game_over = self.game.is_game_over()
        return self.game.board, reward, done, game_over, {}

    def reset(self):
        self.game = Game2048()
        return self.game.board

    def render(self):
        print(self.game.board)

if __name__ == "__main__":
    env = Game2048_env()
    state = env.reset()
    env.render()

    done = False
    game_over = False
    while not game_over:
        # action = int(input("Scegli un'azione (0: sinistra, 1: sopra, 2: destra, 3: sotto): "))
        action = np.random.randint(0,3)
        print(action)
        state, reward, done, game_over, _ = env.step(action)
        env.render()
        print(f"Reward: {reward}")

    print("Game Over!")
