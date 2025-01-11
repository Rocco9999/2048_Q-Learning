import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.optimizers import SGD
from collections import deque
import random

class DQNModel:
    def __init__(self, state_shape, action_space, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='huber')
        return model

class DQNAgent:
    def __init__(self, state_shape, action_space, gamma=0.90, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.000009, batch_size=256, memory_size=100000):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.priorities = deque(maxlen=memory_size)
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
        #print("Sto salvando in memory gli stati")
        self.memory.append((state, action, reward, next_state, done))
        initial_priority = 1.0  # Priorità alta per nuove esperienze
        self.priorities.append(initial_priority)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        #print("Inizio replay")
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([transition[0] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])

        # Ottieni previsioni in un solo passaggio
        q_values_current = self.model.predict(states, verbose=0)
        q_values_next_target = self.target_model.predict(next_states, verbose=0)

        # Prepara i target
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                q_values_current[i][action] = reward
            else:
                # Double DQN update: usa l'argmax del modello primario e Q-value del modello target
                best_action = np.argmax(q_values_next_target[i])
                q_values_current[i][action] = reward + self.gamma * q_values_next_target[i][best_action]

        history = self.model.fit(states, q_values_current, batch_size=self.batch_size, verbose=0)
        #print("Addestro il modello vero e proprio")

        self.loss_history.append(history.history['loss'][0])

        if self.epsilon > self.epsilon_min:
            if self.step_counter % 1000 == 0:
                self.epsilon *= 0.9
        
        self.step_counter += 1
        if(self.step_counter % 100 == 0):
            self.update_target_model()
            #print(f"Target model aggiornato al passo {self.step_counter}")

        if self.step_counter % 10000 == 0 and self.step_counter >= len(self.memory) + int(len(self.memory) * 0.1):
            self.clean_memory(percentage_to_remove=0.1)
            print(f"Mmeoria pulita al passo {self.step_counter}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()

    def save_model(self, path):
        self.model.save(path)

    def clean_memory(self, percentage_to_remove):
        """Rimuove una percentuale delle esperienze meno recenti dal buffer."""
        num_to_remove = int(len(self.memory) * percentage_to_remove)
        if num_to_remove > 0:
            print(f"Rimuovo {num_to_remove} esperienze meno recenti dal buffer.")
            for _ in range(num_to_remove):
                self.memory.popleft()
