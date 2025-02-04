import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Dropout, Conv2D, ReLU, Concatenate, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from collections import deque
import random
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, Callback
from rl.memory import SequentialMemory, RingBuffer
# from memory_profiler import profile
# import gc
import os
import pickle


"""Questo modello utilizza una rete a 198 milioni di parametri conshape 16 4 4, non utilizza il fit ma 
calcola la loss direttamente la i valori q predetti e target, calcola il gradiente e propaga nella rete.
Funziona con agente maindqlcnn step2. Inoltre limita il salvataggio in memoria a un massimo di due elementi
uguali consecutivi"""

class PrioritizedSequentialMemory(SequentialMemory):
    def __init__(self, limit, alpha=0.0, **kwargs):
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

    def sample(self, batch_size, beta=1.0):
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
    
    def get_last(self):
        if self.nb_entries > 0:
            return self._make_full_transition(self.nb_entries - 1)
        return None

    def get_third_last(self):
        if self.nb_entries > 1:
            return self._make_full_transition(self.nb_entries - 2)
        return None
    
    def clean_low_score_episodes(self, n_to_remove=10):
        """
        Pulisce il buffer rimuovendo gli episodi con lo score (somma delle reward positive) più basso.

        Args:
            memory (PrioritizedSequentialMemory): Il buffer delle transizioni.
            n_to_remove (int): Numero di episodi da rimuovere.

        Returns:
            None: Modifica il buffer in-place.
        """
        # Raggruppa le transizioni in episodi
        episodes = []
        current_episode = []
        episode_indices = []  # Salva gli indici delle transizioni
        current_indices = []

        for i in range(self.nb_entries):
            transition = self._make_full_transition(i)
            current_episode.append(transition)
            current_indices.append(i)

            if transition[4]:  # Se `done=True`, termina l'episodio
                episodes.append(current_episode)
                episode_indices.append(current_indices)
                current_episode = []
                current_indices = []

        # Calcola lo score degli episodi come somma delle reward positive
        episode_scores = [
            sum(t[2] for t in ep if t[2] > 0) for ep in episodes
        ]

        # Trova gli indici degli episodi con il punteggio più basso
        worst_episode_indices = np.argsort(episode_scores)[:n_to_remove]
        print(f"Indici da rimuovere: {worst_episode_indices}")

        #No shuffle
        indices_to_keep = set(range(self.nb_entries))
        for idx in worst_episode_indices:
            indices_to_keep -= set(episode_indices[idx])


        #DECOMMENTARE PER MISCHIARE
        # worst_episode_indices_set = set(worst_episode_indices)
        # good_episode_indices = []
        # for i, ep_idx in enumerate(episode_indices):
        #     if i not in worst_episode_indices_set:
        #         good_episode_indices.append(ep_idx)

        # # 3) Mescola i blocchi di episodi come unità
        # random.shuffle(good_episode_indices)
        # indices_to_keep = []
        # for ep_idx in good_episode_indices:
        #     for i_orig in ep_idx:
        #         indices_to_keep.append(i_orig)


        # Ricostruisci il buffer mantenendo solo gli indici rimanenti
        new_observations = [self.observations[i] for i in indices_to_keep]
        new_actions = [self.actions[i] for i in indices_to_keep]
        new_rewards = [self.rewards[i] for i in indices_to_keep]
        new_terminals = [self.terminals[i] for i in indices_to_keep]

        self.observations = RingBuffer(self.limit)
        self.actions = RingBuffer(self.limit)
        self.rewards = RingBuffer(self.limit)
        self.terminals = RingBuffer(self.limit)

        # Reinserisci le transizioni filtrate
        for obs, act, rew, term in zip(new_observations, new_actions, new_rewards, new_terminals):
            super().append(observation=obs, action=act, reward=rew, terminal=term)

        print(f"Lunghezza memoria: {self.nb_entries}")

        # Ricostruisci `self.priorities` con la dimensione originale
        new_priorities = np.zeros(self.limit, dtype=np.float32)  # Ritorna alla dimensione originale
        for i, index in enumerate(indices_to_keep):
            new_priorities[index] = self.priorities[i]  # Mappa i valori vecchi ai nuovi indici
        self.priorities = new_priorities

        self.max_priority = np.max(self.priorities) if np.any(self.priorities) else 1.0

class DQNModel:
    def __init__(self, state_shape, action_space, learning_rate=5e-5):
        self.state_shape = state_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(16, 4, 4))
    
        # Blocchi convolutivi
        x = self.conv_block(inputs, output_dim=2048)
        x = ReLU()(x)
        x = self.conv_block(x, output_dim=2048)
        x = ReLU()(x)
        x = self.conv_block(x, output_dim=2048)
        x = ReLU()(x)
        
        # Flatten per passare ai livelli completamente connessi
        x = Flatten()(x)
        
        # Livelli completamente connessi
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)  # Dropout per ridurre l'overfitting
        outputs = Dense(self.action_space, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        model.summary()
        return model
    

    def conv_block(self, input_tensor, output_dim):
        """
        Implementa un blocco convolutivo che utilizza kernel di diverse dimensioni.
        """
        d = output_dim // 4  # Divide il numero di filtri in 4
        conv1 = Conv2D(d, kernel_size=1, padding='same')(input_tensor)
        conv2 = Conv2D(d, kernel_size=2, padding='same')(input_tensor)
        conv3 = Conv2D(d, kernel_size=3, padding='same')(input_tensor)
        conv4 = Conv2D(d, kernel_size=4, padding='same')(input_tensor)

        # Concatena i risultati delle convoluzioni
        concatenated = Concatenate()([conv1, conv2, conv3, conv4])
        return ReLU()(concatenated)

class DQNAgent:
    def __init__(self, state_shape, action_space, gamma=0.99, decay_episodes = 200, epsilon=0.9, epsilon_min=0.001, epsilon_decay=0.9999, batch_size=64, memory_size=50000, alpha=0.0, beta=1.0, beta_increment=1e-5):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
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
        self.change_lr = False

    def encode_state(self, board):
        # One-hot encode il valore log2 del board
        board_flat = [0 if e == 0 else int(np.log2(e)) for e in board.flatten()]
        one_hot_encoded = tf.one_hot(board_flat, depth=16)  # 16 canali
        # Reshape per ottenere la forma (1, 16, 4, 4)
        encoded = tf.reshape(one_hot_encoded, (1, 4, 4, 16))
        return tf.transpose(encoded, perm=[0, 3, 1, 2])


    def same_move(self, state, next_state, last_memory):
        return np.array_equal(state, last_memory[0]) and np.array_equal(next_state, last_memory[3])

    def remember(self, state, action, reward, done, next_state):
        if done and np.max(state) >= 1024:
            self.change_lr = True
        
        processed_state = self.encode_state(state)
        processed_next_state = self.encode_state(next_state)
        if self.memory.nb_entries < 3:
            self.memory.append(processed_state, action, reward, done)
            return True
        else:
            is_equal = self.same_move(processed_state, processed_next_state, self.memory.get_third_last())
            if done or not is_equal:
                self.memory.append(processed_state, action, reward, done)
                return True
            return False

    def change_lr_function(self):
        current_lr = self.model.optimizer.learning_rate.numpy()
        if self.change_lr == True:
            new_lr = max(current_lr * 0.98, 1e-6)  # Evitiamo che scenda sotto 1e-6
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(f"Learning Rate aggiornato a: {new_lr:.8f}")
            self.change_lr = False
            return new_lr
        
        self.change_lr = False
        return current_lr


    def act(self, state):
        self.update_epsilon()

        #Incrementiamo il numero degli step
        self.step_counter += 1

        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)
        
        processed_state = self.encode_state(state)
        q_values = self.model.predict(processed_state, verbose=0)
        max_q_value = np.argmax(q_values[0])
        return max_q_value

    def act_ripetitive(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(legal_moves) if legal_moves else np.random.randint(self.action_space)
        
        processed_state = self.encode_state(state)
        q_values = self.model.predict(processed_state, verbose=0)
        if legal_moves:  # Se ci sono mosse valide
            max_q_value = np.argmax([q_values[0][action] for action in legal_moves])
            return legal_moves[max_q_value]
        
        return np.argmax(q_values[0])

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def update_epsilon(self):
        self.epsilon = max( self.epsilon_min, self.epsilon_start * (self.epsilon_decay ** self.step_counter))
        #self.beta = min(1.0, self.beta + self.beta_increment)

    def debug_priorities(self):
        print("Priorità nel buffer:")
        valid_slice = self.memory.priorities[:self.memory.nb_entries]
        print(valid_slice)

    
    def replay(self, episode):
        #print("Inizio replay")
        if self.memory.nb_entries < self.batch_size or self.epsilon >= 1:
            return
        
        # Campionamento dal buffer prioritario
        batch, indices, _ = self.memory.sample(self.batch_size, beta=self.beta)

        states      = np.array([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch])
        rewards     = np.array([b[2] for b in batch])
        new_states = np.array([b[3] for b in batch])
        terminals       = np.array([b[4] for b in batch], dtype=bool)

        # Previsione per gli stati correnti e successivi
        with tf.GradientTape() as tape:
            q_values = self.model(np.squeeze(states, axis=1), training=True)  # Q(s, a)
            next_q_values = self.target_model(np.squeeze(new_states, axis=1), training=False)  # Q(s', a')

            # Calcola i target Q
            targets = q_values.numpy()  # Copia dei Q-values
            for i in range(self.batch_size):
                if terminals[i]:
                    targets[i, actions[i]] = rewards[i]  # Reward per stati terminali
                else:
                    targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

            q_values_for_actions = tf.gather(q_values, actions, axis=1, batch_dims=1)
            # Loss MSE
            loss = tf.reduce_mean(tf.square(targets - q_values))

        # Aggiorna i pesi
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # Salva la perdita
        self.loss_history.append(float(loss))

        # Aggiorna le priorità nel replay buffer (se usi Prioritized Experience Replay)
        td_errors = tf.abs(tf.gather(targets, actions, axis=1, batch_dims=1) - q_values_for_actions)
        self.memory.update_priorities(indices, td_errors.numpy())


        # ADDESTRAMENTO CON METOFO FIT DI TENSORFLOW, MOLTO PIù LENTO
        # td_errors = tf.abs(targets - q_values).max(axis=1)
        # self.memory.update_priorities(indices, td_errors)
        
        # history = self.model.fit(states, targets, epochs=1, verbose=0)
        # self.loss_history.append(history.history['loss'][0])
        # #print("Addestro il modello vero e proprio")


    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()
        print("Modello caricato")

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
    

    def load_memory(self, path):
        # Carica il buffer di memoria
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                self.memory = pickle.load(f)
            print("Memoria caricata.")
        else:
            print("File di memoria non trovato. Inizializzazione di un nuovo buffer.")

        print("Cancella memoria")
        if self.memory.nb_entries > self.batch_size:
                self.memory.clean_low_score_episodes(n_to_remove=99)
        print("Memoria cancellata")



    def save_agent_state_checkpoint(self, directory, episode, arrays, checkpoint_name):
        """Salva lo stato completo dell'agente (modello, memoria, variabili)."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Salva il modello
        model_path = os.path.join(directory, f"model_{checkpoint_name}.h5")
        self.save_model(model_path)

        # Salva il buffer di memoria usando pickle
        memory_path = os.path.join(directory, f"memory_{checkpoint_name}.pkl")
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
        variables_path = os.path.join(directory, f"variables_{checkpoint_name}.pkl")
        with open(variables_path, 'wb') as f:
            pickle.dump(variables, f)

        arrays_path = os.path.join(directory, f"arrays_{checkpoint_name}.pkl")
        with open(arrays_path, 'wb') as f:
            pickle.dump(arrays, f)

        print(f"Stato dell'agente salvato in: {directory} (episodio {episode})")

    def load_agent_state_checkpoint(self, directory, checkpoint_name):
        """Carica lo stato completo dell'agente (modello, memoria, variabili)."""
        # Carica il modello
        model_path = os.path.join(directory, f"model_{checkpoint_name}.h5")
        if os.path.isfile(model_path):
            self.load_model(model_path)
            print("Modello caricato.")
        else:
            print("File del modello non trovato. Inizializzazione di un nuovo modello.")

        # Carica il buffer di memoria
        memory_path = os.path.join(directory, f"memory_{checkpoint_name}.pkl")
        if os.path.isfile(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)
            print("Memoria caricata.")
        else:
            print("File di memoria non trovato. Inizializzazione di un nuovo buffer.")

        # Carica le altre variabili
        variables_path = os.path.join(directory, f"variables_{checkpoint_name}.pkl")
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

        arrays_path = os.path.join(directory, f"arrays_{checkpoint_name}.pkl")
        if os.path.isfile(arrays_path):
            with open(arrays_path, 'rb') as f:
                arrays = pickle.load(f)
            max_tile_list = arrays['max_tile_list']
            score_list = arrays['score_list']
            loss_history = arrays['loss_history']
            episode = arrays['episode']
            # Carica altre variabili se necessario
            print("Variabili caricate.")
        else:
            print("File delle variabili non trovato. Inizializzazione con valori predefiniti.")

        return episode, max_tile_list, score_list, loss_history