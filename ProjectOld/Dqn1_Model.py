import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
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
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

class DQNAgent:
    def __init__(self, state_shape, action_space, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.000009, batch_size=128, memory_size=10000):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = DQNModel(state_shape, action_space).model
        self.target_model = DQNModel(state_shape, action_space).model
        self.update_target_model()
        self.step_counter = 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        #print("Sto salvando in memory gli stati")
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #print("Sto prendendo una decisione su come muovermi")
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        #print(q_values)
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
        q_values_next_model = self.model.predict(next_states, verbose=0)
        q_values_next_target = self.target_model.predict(next_states, verbose=0)

        # Prepara i target
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                q_values_current[i][action] = reward
            else:
                # Double DQN update: usa l'argmax del modello primario e Q-value del modello target
                best_action = np.argmax(q_values_next_model[i])
                q_values_current[i][action] = reward + self.gamma * q_values_next_target[i][best_action]

        self.model.fit(states, q_values_current, batch_size=self.batch_size, verbose=0)
        #print("Addestro il modello vero e proprio")

        if self.epsilon > self.epsilon_min:
            #self.epsilon *= self.epsilon_decay
            self.epsilon -= self.epsilon_decay
        
        self.step_counter += 1
        if(self.step_counter % 100 == 0):
            self.update_target_model()
            #print(f"Target model aggiornato al passo {self.step_counter}")

        if self.step_counter % 10000 == 0 and self.step_counter >= len(self.memory) + int(len(self.memory) * 0.1):
            self.clean_memory(percentage_to_remove=0.1)
            print(f"Mmeoria pulita al passo {self.step_counter}")

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target_model()

    def clean_memory(self, percentage_to_remove):
        """Rimuove una percentuale delle esperienze meno recenti dal buffer."""
        num_to_remove = int(len(self.memory) * percentage_to_remove)
        if num_to_remove > 0:
            print(f"Rimuovo {num_to_remove} esperienze meno recenti dal buffer.")
            for _ in range(num_to_remove):
                self.memory.popleft()