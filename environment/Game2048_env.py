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
    rewards_buffer = collections.deque()
    def __init__(self):
        super(Game2048_env, self).__init__()
        self.game = Game2048()
        self.score = 0
        self.penalty = 10
        self.previous_max = 2
        #Osservazioni per l'agente
        self.action_space = spaces.Discrete(4)  # Azioni: sinistra, sopra, destra, sotto
        self.observation_space = spaces.Box(0, 2048, shape=(4, 4), dtype=int)
        self.scaling_factor = 1.2  # Impostato così, ma è pissibile fare uno studio randomico per vedere come ottimizzarlo
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
        
        # Aggiungere un piccolo bonus dinamico se `bonus_progress` è 0
        dynamic_bonus = 0
        if bonus_progress == 0 and valid:
            # Calcola un bonus dinamico basato sul livello corrente
            dynamic_bonus = current_level * 0.1  # Fattore di scala per controllare l'impatto
    

        if not valid:
            # La mossa non sposta nulla
            if game_over:
                # Il gioco è finito
                if max_number in [512, 1024, 2048]:
                    # Reward per un livello massimo significativo
                    reward = bonus_progress + (current_level ** self.scaling_factor)
                else:
                    ratio = max_number / 512
                    penalty = self.calculate_penalty(base_penalty=-10, current_level = current_level)
                    reward += penalty * (1 - ratio)
            else:
                # Mossa non valida ma non è game over
                # Penalità proporzionale allo stato di avanzamento
                penalty = self.calculate_penalty(base_penalty=-5, current_level = current_level)
                reward += penalty
        else:
            # Caso: La mossa è valida
            # Al momento usiamo soltanto lo score come reward
            reward = score

            # Aggiungiamo l'eventuale bonus se abbiamo superato il precedente record
            if bonus_progress > 0:
                reward += bonus_progress
            else:
            # Aggiungere il bonus dinamico
                reward += dynamic_bonus

        if max_number >= 512:
            reward += (current_level ** self.scaling_factor) * 2
            #Reward per quando supera il 512
        
        # Normalizziamo la reward
        normalized_reward = self.normalizer.update_and_normalize(reward)

        return normalized_reward


    def reset(self):
        #Resetta l'ambiente di gioco dopo aver terminato
        self.game = Game2048()
        return self.game.board

    def showMatrix(self):
        print(self.score)
        print(self.game.board)

class AdaptiveRewardNormalizer:
    def __init__(self, min_size=10, cleaning_threshold = 1.5):
        # Valori iniziali di fallback
        self.min_size = min_size
        self.min_val = -10
        self.max_val = 2048
        self.cleaning_threshold = cleaning_threshold

    def pulisci_buffer(self, rewards_buffer):
        # Suddividi il buffer in blocchi
        if len(rewards_buffer) < self.min_size:
            return rewards_buffer  # Nessuna pulizia necessaria
        
        rewards_list = list(rewards_buffer)
        # Calcola il primo 90% del buffer
        first_90_percent = int(len(rewards_list) * 0.90)
        # Calcola la dimensione dei blocchi (un decimo del primo 90%)
        if first_90_percent > 0:
            numRange = max(1, math.ceil(first_90_percent / 10))
        else:
            numRange = 1
        blocks = [rewards_list[i:i + numRange] for i in range(0, first_90_percent, numRange)]
        variances = [np.var(block) for block in blocks]
        
        # Calcola la varianza media e la soglia dinamica
        mean_variance = np.mean(variances)
        std_variance = np.std(variances)
        self.cleaning_threshold = mean_variance + 2.0 * std_variance

        num_anomalous_blocks = sum(1 for var in variances if var > self.cleaning_threshold)
        if num_anomalous_blocks < len(blocks) * 0.1:  # Rimuovi solo se più del 10% dei blocchi è anomalo
            return rewards_buffer

        # Identifica i blocchi con varianze anomale
        indices_to_remove = set()
        for block_index, var in enumerate(variances):
            if var > self.cleaning_threshold:
                start_idx = block_index * numRange
                end_idx = min((block_index + 1) * numRange, first_90_percent)
                indices_to_remove.update(range(start_idx, end_idx))

        # Rimuovi gli elementi anomali
        cleaned_rewards = [v for i, v in enumerate(rewards_list) if i not in indices_to_remove]
        rewards_buffer = collections.deque(cleaned_rewards)

        # Log per il debug
        # print(f"Buffer size before cleaning: {len(rewards_list)}")
        # print(f"Mean variance: {mean_variance:.4f}, Std variance: {std_variance:.4f}")
        # print(f"Cleaning threshold: {self.cleaning_threshold:.4f}")
        # print(f"Removed {len(indices_to_remove)} rewards from buffer.")
        # print(f"New buffer size: {len(rewards_buffer)}")

        return rewards_buffer
    
    def update_and_normalize(self, reward, rewards_buffer):
        # Aggiorna il buffer con la nuova reward
        rewards_buffer.append(reward)

        #ORA AGGIORNIAMO LA GRANDEZZA DEL BUFFER
        variance = np.var(rewards_buffer)
        # Calcola la varianza
        #print("La varianza è: ", variance)
        #print("Il buffer è lungo: ", len(rewards_buffer))

        # Se non abbiamo abbastanza dati, usiamo i fallback (range statico)
        if len(rewards_buffer) < 10:
            normalize_static = self._normalize_static(reward)
            return normalize_static, rewards_buffer
        
        # Pulisci il buffer se necessario
        rewards_buffer = self.pulisci_buffer(rewards_buffer)

        # Calcoliamo il min e max dal buffer
        current_min = min(rewards_buffer)
        current_max = max(rewards_buffer)

        # Normalizzazione dinamica
        if current_max > current_min:
            normalized_reward = self._dynamic_asymmetric_normalize(reward, current_min, current_max)
        else:
            normalized_reward = 0
        
        return normalized_reward, rewards_buffer

    def _dynamic_asymmetric_normalize(self, reward, current_min, current_max):
        if current_min > 0:
            # Tutte le reward sono positive
            if current_max == 0:
                return 0
            else:
                return min(reward/current_max,1)

        if current_max < 0:
            # Tutte le reward sono negative
            return -1 + (reward - current_min)/(-current_min)

        # Caso generale: abbiamo sia valori negativi che non negativi nel buffer
        if reward < 0:
            return -1 + (reward - current_min)/(-current_min)
        else:
            if current_max == 0:
                return 0
            else:
                return min(reward/current_max,1)


    def _normalize_static(self, reward):
        #Qui utilizziamo valori standard
        if reward < 0:
            return -1 + (reward - self.min_val)/(-self.min_val)
        else:
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
        #print("L'azione scelta è stata: ", action)
        state, reward, done, max_number = env.step(action)
        #Ci salviamo anche lo stato in vista dell'interazione con l'agente
        env.showMatrix()
        print(f"Il numero più alto in griglia è: {max_number}")
        print(f"Reward: {reward}")

    #print("Numero di mosse: ", numAction)
    print("Game Over!")
