import gymnasium as gym
import numpy as np
from gymnasium import spaces
from math import log2
import collections
import math
import tkinter as tk
from tkinter import Frame, Label, CENTER, Button
import tensorflow as tf
import numpy as np
from keras.models import Sequential, Model


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
        self.prev_max_tile = 2
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
        reward = self.calculate_reward2(score=score, valid=valid, game_over=game_over, max_number=max_number)

        if game_over:
            done = True

        return self.game.moved_board, reward, done, max_number, valid
    
    def calculate_reward2(self, score, valid, game_over, max_number):
        reward = 0

        if not valid and not game_over:
            reward = -10
        else:
            reward = score

        return reward


    def reset(self):
        #Resetta l'ambiente di gioco dopo aver terminato
        self.game = Game2048()
        self.score = 0
        self.prev_max_tile = 2
        return self.game.board

    def showMatrix(self):
        print(self.score)
        print(self.game.board)


class Game2048_GUI:
    def __init__(self, root, env, auto = None):
        self.env = env
        self.root = root
        self.root.title("2048 - Tkinter")
        self.root.geometry("700x700")
        self.grid_cells = []
        self.num_moves = 0
        self.last_action = None
        # Mostra il menu di selezione modalità
        if auto is None:
            self.show_mode_selection()

    def show_mode_selection(self):
        # Finestra per la selezione della modalità
        mode_frame = Frame(self.root, bg="#92877d")
        mode_frame.pack(expand=True)

        Label(
            mode_frame,
            text="Seleziona la modalità di gioco",
            font=("Helvetica", 16),
            bg="#92877d",
            fg="white",
        ).pack(pady=20)

        # Pulsante per modalità manuale
        Button(
            mode_frame,
            text="Manuale (Usa le frecce)",
            font=("Helvetica", 14),
            command=lambda: self.start_game(False, mode_frame),
            width=20,
        ).pack(pady=10)

        # Pulsante per modalità automatica
        Button(
            mode_frame,
            text="Automatica (Mosse random)",
            font=("Helvetica", 14),
            command=lambda: self.start_game(True, mode_frame),
            width=20,
        ).pack(pady=10)

        # Pulsante per modalità con IA
        Button(
            mode_frame,
            text="Automatico (Modello dqn)",
            font=("Helvetica", 14),
            command=lambda: self.start_game(None, mode_frame),
            width=20,
        ).pack(pady=10)

    def start_game(self, auto, mode_frame):
        # Rimuove il frame della selezione modalità e inizia il gioco
        mode_frame.destroy()
        self.init_grid()
        self.update_grid()
        if auto is True:
            self.auto_play()
        elif auto is False:
            self.root.bind_all("<Key>", self.key_pressed)
        elif auto is None:
            self.model = tf.keras.models.load_model("dqn_model_2048_2048_1858.h5")
            print("Modello caricato")
            self.model_play()

    def init_grid(self):
        # Aggiungi un frame per il punteggio
        score_frame = Frame(self.root, bg="#92877d")
        score_frame.grid(row=0, column=0, columnspan=4, pady=10)

        # Etichette per il punteggio
        self.score_label = Label(score_frame, text="Score: 0", font=("Helvetica", 16), bg="#92877d", fg="white")
        self.score_label.grid(row=0, column=0, padx=5)

        self.moves_label = Label(score_frame, text="Moves: 0", font=("Helvetica", 16), bg="#92877d", fg="white")
        self.moves_label.grid(row=0, column=1, padx=5)

        self.last_action_label = Label(score_frame, text="Last Move: None", font=("Helvetica", 16), bg="#92877d", fg="white")
        self.last_action_label.grid(row=0, column=2, padx=5)

        # Etichetta per la griglia
        grid_frame = Frame(self.root, bg="#92877d", width=700, height=700)
        grid_frame.grid(row=1, column=0, columnspan=4)

        for i in range(4):
            row = []
            for j in range(4):
                cell = Frame(grid_frame, bg="#9e948a", width=700, height=700)
                cell.grid(row=i, column=j, padx=5, pady=5)
                label = Label(cell, text="", font=("Helvetica", 20), width=5, height=2)
                label.grid()
                row.append(label)
            self.grid_cells.append(row)

    def update_grid(self):
        self.score_label.config(text=f"Score: {self.env.score}")  # Aggiorna lo score
        self.moves_label.config(text=f"Moves: {self.num_moves}")  # Aggiorna il numero di mosse
        self.last_action_label.config(text=f"Last Move: {self.last_action}")  # Aggiorna l'ultima mossa
        for i in range(4):
            for j in range(4):
                number = self.env.game.board[i][j]
                if number == 0:
                    self.grid_cells[i][j].configure(text="", bg="#9e948a")
                else:
                    self.grid_cells[i][j].configure(
                        text=str(number),
                        bg=self.get_tile_color(number),
                        fg="#f9f6f2" if number > 4 else "#776e65"
                    )
        self.root.update_idletasks()

    def key_pressed(self, event):
        print(f"Tasto premuto: {event.keysym}")  # Debug
        key_map = {"Left": 0, "Up": 1, "Right": 2, "Down": 3}
        if event.keysym in key_map:
            action = key_map[event.keysym]
            self.last_action = event.keysym  # Registra l'azione
            self.num_moves += 1  # Incrementa il contatore delle mosse
            state, reward, done, max_number, moved = env.step(action)
            self.env.game.board = state
            self.update_grid()
            if done:
                self.show_game_over()


    def auto_play(self):
        # Genera un'azione casuale
        action_map = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
        action = np.random.randint(0, 4)
        self.last_action = action_map[action]  # Registra l'azione
        self.num_moves += 1  # Incrementa il contatore delle mosse
        state, reward, done, max_number, moved = self.env.step(action)
        self.env.game.board = state
        self.update_grid()
        if done:
            self.show_game_over()
            return  # Termina il gioco
        # Pianifica la prossima mossa dopo 500 ms
        self.root.after(500, self.auto_play)


    def model_play(self):
        # Genera un'azione casuale
        action_map = {0: "Left", 1: "Up", 2: "Right", 3: "Down"}
        state = self.env.game.board.copy()
        processed_state = self.encode_state(state)
        q_values = self.model.predict(processed_state, verbose=0)
        max_q_value = np.argmax(q_values[0])
        legal_moves = []
        for act in range(4):
            # Check if the move is legal
            leagal, _= self.env.game.move(act, trial=True)
            if leagal:
                legal_moves.append(act)
        
        if legal_moves:  # Se ci sono mosse valide
            max_q_value = np.argmax([q_values[0][a] for a in legal_moves])
        
        action = legal_moves[max_q_value]
        
        self.last_action = action_map[action]  # Registra l'azione
        self.num_moves += 1  # Incrementa il contatore delle mosse
        state, reward, done, max_number, moved = self.env.step(action)
        self.env.game.board = state
        self.update_grid()
        if done:
            self.show_game_over()
            return  # Termina il gioco
        # Pianifica la prossima mossa dopo 500 ms
        self.root.after(500, self.model_play)

    def show_game_over(self):
        self.root.unbind("<Key>")
        game_over_label = Label(self.root, text="Game Over!", font=("Helvetica", 30), bg="#92877d", fg="white")
        game_over_label.place(relx=0.5, rely=0.5, anchor=CENTER)

    @staticmethod
    def get_tile_color(value):
        colors = {
            2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
            32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
            512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
        }
        return colors.get(value, "#3c3a32")


    def encode_state(self, board):
        # One-hot encode il valore log2 del board
        board_flat = [0 if e == 0 else int(np.log2(e)) for e in board.flatten()]
        one_hot_encoded = tf.one_hot(board_flat, depth=16)  # 16 canali
        # Reshape per ottenere la forma (1, 16, 4, 4)
        encoded = tf.reshape(one_hot_encoded, (1, 4, 4, 16))
        return tf.transpose(encoded, perm=[0, 3, 1, 2])


# Esegui l'interfaccia grafica
if __name__ == "__main__":
    root = tk.Tk()
    env = Game2048_env()
    gui = Game2048_GUI(root, env)
    root.mainloop()
