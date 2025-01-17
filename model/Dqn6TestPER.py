import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.optimizers import SGD
from collections import deque
import random
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, Callback
from rl.memory import SequentialMemory
# from memory_profiler import profile
# import gc
import os
import pickle

class PrioritizedSequentialMemory(SequentialMemory):
    def __init__(self, limit, alpha=0.6, **kwargs):
        super().__init__(limit, **kwargs)
        self.alpha = alpha  # Controlla la forza della priorità

        # Vettore delle priorità, dimensione = limit
        self.priorities = np.zeros((limit,), dtype=np.float32)
        self.max_priority = 1.0  # Priorità iniziale

    def append(self, observation, action, reward, terminal=False, training=True, **kwargs):
        """
        Sovrascrivi append: keras-rl di solito si aspetta i parametri (observation, action, reward, terminal, ...)
        Non passiamo next_state qui, perché SequentialMemory lo gestisce tra forward/backward.
        """
        super().append(observation=observation,
                       action=action,
                       reward=reward,
                       terminal=terminal,
                       training=training,
                       **kwargs)

        # Aggiorniamo la priorità della transizione appena inserita
        current_index = (self.nb_entries - 1) % self.limit
        self.priorities[current_index] = self.max_priority

    def _make_full_transition(self, index):
        """
        Ricostruisce (state0, action, reward, next_state, done) 
        usando i vettori: observations, actions, rewards, terminals.
        Per semplicità, assumiamo window_length=1.
        """
        state0 = self.observations[index]
        action = self.actions[index]
        reward = self.rewards[index]
        done   = self.terminals[index]

        # next_state se non è done e c'è un passo successivo
        if (not done) and (index < self.nb_entries - 1):
            next_state = self.observations[index + 1]
        else:
            next_state = np.zeros_like(state0)

        return (state0, action, reward, next_state, done)

    def sample(self, batch_size, beta=0.4):
        """
        Campiona 'batch_size' transizioni PER.
         Ritorna: (batch, indices, is_weights)
          - batch: lista di tuple (state0, action, reward, next_state, done)
        """
        if self.nb_entries == 0:
            raise ValueError("Replay buffer vuoto.")

        # Consideriamo solo gli indici [0..nb_entries-1]
        valid_priorities = self.priorities[:self.nb_entries] ** self.alpha
        sum_priorities = valid_priorities.sum()
        if sum_priorities == 0:
            # Evita division by zero se le priorità sono 0
            probabilities = np.ones(self.nb_entries, dtype=np.float32) / self.nb_entries
        else:
            probabilities = valid_priorities / sum_priorities

        # Campioniamo in base alle probabilità
        indices = np.random.choice(self.nb_entries, size=batch_size, p=probabilities)

        # Ricostruisci la transizione
        batch = [self._make_full_transition(idx) for idx in indices]

        # Calcolo dei pesi di importanza
        is_weights = (self.nb_entries * probabilities[indices]) ** -beta
        is_weights /= is_weights.max()

        return batch, indices, is_weights

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        """
        Aggiorna la priorità in base all'errore TD.
        """
        for index, td_error in zip(indices, td_errors):
            p = abs(td_error) + epsilon
            self.priorities[index] = p
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.nb_entries


class DQNModel:
    def __init__(self, state_shape, action_space, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer=SGD(learning_rate=self.learning_rate), loss='huber')
        return model

class DQNAgent:
    def __init__(self, state_shape, action_space, gamma=0.90, decay_episodes = 200, epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.995, batch_size=200, memory_size=4000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_episodes = decay_episodes
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay1 = (epsilon - epsilon_min) / decay_episodes
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = PrioritizedSequentialMemory(limit=memory_size, alpha=alpha, window_length=1)
        self.beta = beta
        self.beta_increment = beta_increment
        self.model = DQNModel(state_shape, action_space).model
        self.target_model = DQNModel(state_shape, action_space).model
        self.update_target_model()
        self.step_counter = 0
        self.loss_history = []
        self.lr_min = 0.00025


    def remember(self, state, action, reward, done):
        self.memory.append(state, action, reward, done)

    def act(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_moves) if legal_moves else np.random.randint(self.action_space)
        
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        if legal_moves:  # Se ci sono mosse valide
            max_q_value = np.argmax([q_values[0][action] for action in legal_moves])
            return legal_moves[max_q_value]
        
        return np.argmax(q_values[0])

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay1
            self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def debug_priorities(self):
        print("Priorità nel buffer:")
        valid_slice = self.memory.priorities[:self.memory.nb_entries]
        print(valid_slice)

    
    def replay(self, episode):
        #print("Inizio replay")
        if self.memory.nb_entries < self.batch_size or self.epsilon >= 1:
            return
        #DISATTIVIAMO LA PER DOPO 200 EPISODI
        if episode > 200:
            self.memory.alpha = 0
        # Campionamento dal buffer prioritario
        batch, indices, is_weights = self.memory.sample(self.batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch])
        rewards     = np.array([b[2] for b in batch])
        new_states = np.array([b[3] for b in batch])
        terminals       = np.array([b[4] for b in batch], dtype=bool)
        new_states = np.squeeze(new_states)

        # Ottieni previsioni in un solo passaggio
        states = states.reshape(self.batch_size, 4, 4)
        new_states = new_states.reshape(self.batch_size, 4, 4)

        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(new_states, verbose=0)
        targets = q_values.copy()

        for i in range(self.batch_size):
            target = rewards[i]
            if not terminals[i]:
                target += self.gamma * np.max(next_q_values[i])
            targets[i, actions[i]] = target

        # Aggiorna il modello
        td_errors = np.abs(targets - q_values).max(axis=1)
        self.memory.update_priorities(indices, td_errors)

        history = self.model.fit(states, targets, epochs=1, sample_weight=is_weights, verbose=0)
        #print("Addestro il modello vero e proprio")

        self.loss_history.append(history.history['loss'][0])
        
        self.step_counter += 1
        if(self.step_counter % 500 == 0):
            print("Aggiorno il modello target")
            self.update_target_model()
            print(f"Target model aggiornato al passo {self.step_counter}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()

    def save_model(self, path):
        self.model.save(path)

    def save_agent_state(self, directory, episode, arrays):
        """Salva lo stato completo dell'agente (modello, memoria, variabili)."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Salva il modello
        model_path = os.path.join(directory, f"model_episode_{episode}.h5")
        self.save_model(model_path)

        # Salva il buffer di memoria usando pickle
        memory_path = os.path.join(directory, f"memory_episode_{episode}.pkl")
        with open(memory_path, 'wb') as f:
            pickle.dump(self.memory, f)

        # Salva le altre variabili
        variables = {
            'epsilon': self.epsilon,
            'beta': self.beta,
            'step_counter': self.step_counter,
            'loss_history': self.loss_history,
            # Aggiungi altre variabili se necessario
        }
        variables_path = os.path.join(directory, f"variables_episode_{episode}.pkl")
        with open(variables_path, 'wb') as f:
            pickle.dump(variables, f)

        arrays_path = os.path.join(directory, f"arrays_episode_{episode}.pkl")
        with open(arrays_path, 'wb') as f:
            pickle.dump(arrays, f)

        print(f"Stato dell'agente salvato in: {directory} (episodio {episode})")

    def load_agent_state(self, directory, episode):
        """Carica lo stato completo dell'agente (modello, memoria, variabili)."""
        # Carica il modello
        model_path = os.path.join(directory, f"model_episode_{episode}.h5")
        if os.path.isfile(model_path):
            self.load_model(model_path)
            print("Modello caricato.")
        else:
            print("File del modello non trovato. Inizializzazione di un nuovo modello.")

        # Carica il buffer di memoria
        memory_path = os.path.join(directory, f"memory_episode_{episode}.pkl")
        if os.path.isfile(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
            print("Memoria caricata.")
        else:
            print("File di memoria non trovato. Inizializzazione di un nuovo buffer.")

        # Carica le altre variabili
        variables_path = os.path.join(directory, f"variables_episode_{episode}.pkl")
        if os.path.isfile(variables_path):
            with open(variables_path, 'rb') as f:
                variables = pickle.load(f)
            self.epsilon = variables['epsilon']
            self.beta = variables['beta']
            self.step_counter = variables['step_counter']
            self.loss_history = variables['loss_history']
            # Carica altre variabili se necessario
            print("Variabili caricate.")
        else:
            print("File delle variabili non trovato. Inizializzazione con valori predefiniti.")

        arrays_path = os.path.join(directory, f"arrays_episode_{episode}.pkl")
        if os.path.isfile(arrays_path):
            with open(arrays_path, 'rb') as f:
                arrays = pickle.load(f)
            max_tile_list = arrays['max_tile_list']
            score_list = arrays['score_list']
            loss_history = arrays['loss_history']
            # Carica altre variabili se necessario
            print("Variabili caricate.")
        else:
            print("File delle variabili non trovato. Inizializzazione con valori predefiniti.")

        return episode, max_tile_list, score_list, loss_history