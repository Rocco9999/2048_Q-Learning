import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from collections import deque
import random

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6):
        self.size = size
        self.alpha = alpha
        self.memory = np.zeros(size, dtype=object)
        self.priorities = np.zeros(size, dtype=float)
        self.next_idx = 0

    def add(self, experience, error):
        priority = (error + 1e-5) ** self.alpha
        self.memory[self.next_idx] = experience
        self.priorities[self.next_idx] = priority
        self.next_idx = (self.next_idx + 1) % self.size

    def sample(self, batch_size, beta=0.4):
        scaled_priorities = self.priorities[:len(self.memory)] / sum(self.priorities[:len(self.memory)])
        indices = np.random.choice(len(self.memory), batch_size, p=scaled_priorities)
        importance = (1 / (len(self.memory) * scaled_priorities[indices])) ** beta
        importance /= max(importance)  # Normalizzazione
        return self.memory[indices], indices, importance

    def update_priority(self, indices, errors):
        priorities = (np.maximum(errors, 0.01) + 1e-5) ** self.alpha
        self.priorities[indices] = priorities

    def clean_memory(self, percentage_to_remove=0.1):
        """Rimuove una percentuale delle esperienze meno recenti."""
        num_to_remove = int(self.next_idx * percentage_to_remove)
        if num_to_remove > 0:
            print(f"Rimuovo {num_to_remove} esperienze meno recenti.")
            self.memory[:self.next_idx - num_to_remove] = self.memory[num_to_remove:self.next_idx]
            self.priorities[:self.next_idx - num_to_remove] = self.priorities[num_to_remove:self.next_idx]
            self.next_idx -= num_to_remove

    def __len__(self):
        return self.next_idx

class DQNModel:
    def __init__(self, state_shape, action_space, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0), loss='mse')
        return model

class DQNAgent:
    def __init__(self, state_shape, action_space, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.000009, batch_size=128, memory_size=100000):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = PrioritizedReplayBuffer(memory_size)  # Usa PrioritizedReplayBuffer
        self.model = DQNModel(state_shape, action_space).model
        self.target_model = DQNModel(state_shape, action_space).model
        self.update_target_model()
        self.step_counter = 0


    def update_target_model(self, tau=0.005):
        """Aggiornamento soft del modello target."""
        model_weights = np.array(self.model.get_weights(), dtype=object)
        target_weights = np.array(self.target_model.get_weights(), dtype=object)
        new_weights = tau * model_weights + (1 - tau) * target_weights
        self.target_model.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        """Memorizza esperienze con priorità, calcolando l'errore TD."""
        # Calcola il valore Q corrente per l'azione scelta
        q_values_current = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0][action]

        # Calcola il valore target
        if done:
            target = reward
        else:
            q_values_next = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
            target = reward + self.gamma * np.amax(q_values_next)

        # Calcola l'errore TD
        error = abs(q_values_current - target)

        # Aggiungi l'esperienza al buffer
        self.memory.add((state, action, reward, next_state, done), error)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def replay(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return

        batch, indices, importance = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)

        q_values_current = self.model.predict(states, verbose=0)
        q_values_next_target = self.target_model.predict(next_states, verbose=0)


        targets = np.array(rewards) + self.gamma * np.amax(q_values_next_target, axis=1) * (1 - np.array(dones))
        errors = np.abs(q_values_current[np.arange(self.batch_size), actions] - targets)
        q_values_current[np.arange(self.batch_size), actions] = targets
        
        self.memory.update_priority(indices, errors)
        self.model.fit(states, q_values_current, sample_weight=importance, batch_size=self.batch_size, verbose=0)
        
        if self.epsilon > 0.5:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon -= self.epsilon_decay / 2

        self.step_counter += 1
        if(self.step_counter % 100 == 0):
            self.update_target_model()
            #print(f"Target model aggiornato al passo {self.step_counter}")

        if self.step_counter % 10000 == 0 and self.step_counter >= len(self.memory) + int(len(self.memory) * 0.1):
            self.memory.clean_memory(percentage_to_remove=0.1)
            print(f"Mmeoria pulita al passo {self.step_counter}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()

    def save_model(self, path):
        self.model.save(path)
