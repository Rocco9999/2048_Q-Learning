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

class PrioritizedSequentialMemory(SequentialMemory):
    def __init__(self, limit, alpha=0.6, **kwargs):
        super().__init__(limit, **kwargs)
        self.alpha = alpha  # Controlla la forza della priorità
        self.priorities = np.zeros((limit,), dtype=np.float32)
        self.max_priority = 1.0  # Priorità iniziale alta
        self.index = 0

    def append(self, *args, **kwargs):
        super().append(*args, **kwargs)
        self.priorities[self.index] = self.max_priority
        self.index = (self.index + 1) % self.limit if len(self) == self.limit else len(self) - 1


    def sample(self, batch_size, beta=0.4):
        # Campionamento proporzionale alle priorità
        if len(self) == 0:
            raise ValueError("Replay buffer vuoto.")
        
        priorities = self.priorities[:len(self)] ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self), batch_size, p=probabilities)
        batch = [self._get_single_state(index) for index in indices]

        # Calcola i pesi di importanza
        is_weights = (len(self) * probabilities[indices]) ** -beta
        is_weights /= is_weights.max()

        return batch, indices, is_weights

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        # Aggiorna le priorità basandosi sugli errori TD
        for index, td_error in zip(indices, td_errors):
            self.priorities[index] = np.abs(td_error) + epsilon
        self.max_priority = max(self.priorities[:len(self)])

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
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='huber')
        return model

class DQNAgent:
    def __init__(self, state_shape, action_space, gamma=0.90, decay_episodes = 1500, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.000009, batch_size=256, memory_size=50000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_episodes = decay_episodes
        self.epsilon_decay = (epsilon - epsilon_min) / decay_episodes
        self.batch_size = batch_size
        self.memory = PrioritizedSequentialMemory(limit=memory_size, alpha=alpha, window_length=1)
        self.beta = beta
        self.beta_increment = beta_increment
        self.model = DQNModel(state_shape, action_space).model
        self.target_model = DQNModel(state_shape, action_space).model
        self.update_target_model()
        self.step_counter = 0
        self.loss_history = []


    def update_target_model(self, tau=0.005):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = [
            tau * mw + (1 - tau) * tw
            for mw, tw in zip(model_weights, target_weights)
        ]
        self.target_model.set_weights(new_weights)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def act(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)
        
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        if legal_moves:  # Se ci sono mosse valide
            max_q_value = np.argmax([q_values[0][action] for action in legal_moves])
            return legal_moves[max_q_value]
        
        return np.argmax(q_values[0])
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    
    def replay(self, episode):
        #print("Inizio replay")
        if self.memory.nb_entries < self.batch_size or self.epsilon >= 1:
            for i, transition in enumerate(self.memory.sample(self.memory.nb_entries)):
                print(f"Transizione {i}: {transition}")
            return

        # Campionamento dal buffer prioritario
        batch, indices, is_weights = self.memory.sample(self.batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states = np.array([transition.state0 for transition in batch])
        actions = np.array([transition.action for transition in batch])
        rewards = np.array([transition.reward for transition in batch])
        new_states = np.array([transition.state1 for transition in batch])
        terminals = np.array([transition.terminal1 for transition in batch])
        new_states = np.squeeze(new_states)

        # Ottieni previsioni in un solo passaggio
        states = states.reshape(self.batch_size, 4, 4)
        new_states = new_states.reshape(self.batch_size, 4, 4)

        q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(new_states)
        targets = q_values.copy()

        for i in range(self.batch_size):
            target = rewards[i]
            if not terminals[i]:
                target += self.gamma * np.max(next_q_values[i])
            targets[i, actions[i]] = target

        # Aggiorna il modello
        td_errors = np.abs(targets - q_values).max(axis=1)
        self.memory.update_priorities(indices, td_errors)

        # LOGICA CON TARGET MODEL
        # next_q_values  = self.target_model.predict(new_states, verbose=0)

        # targets[np.arange(self.batch_size), actions] = rewards + (1 - terminals) * self.gamma * np.max(next_q_values, axis=1)

        history = self.model.fit(states, targets, sample_weight=is_weights, verbose=0)
        #print("Addestro il modello vero e proprio")

        self.loss_history.append(history.history['loss'][0])
        
        self.step_counter += 1
        if(self.step_counter % 500 == 0):
            print("Aggiorno il modello target")
            self.update_target_model()
            # print(f"Target model aggiornato al passo {self.step_counter}")

        # if (self.step_counter % 10000 == 0 or len(self.memory) > 50000) and (self.step_counter >= len(self.memory) + int(len(self.memory) * 0.1)):
        #     self.clean_memory(percentage_to_remove=0.1)
        #     print(f"Mmeoria pulita al passo {self.step_counter}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        #self.update_target_model()

    def save_model(self, path):
        self.model.save(path)

    def clean_memory(self, percentage_to_remove):
        """Rimuove una percentuale delle esperienze meno recenti dal buffer."""
        num_to_remove = int(len(self.memory) * percentage_to_remove)
        if num_to_remove > 0:
            print(f"Rimuovo {num_to_remove} esperienze meno recenti dal buffer.")
            for _ in range(num_to_remove):
                self.memory.popleft()
