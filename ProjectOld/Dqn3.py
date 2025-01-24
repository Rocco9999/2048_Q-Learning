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
    def __init__(self, state_shape, action_space, gamma=0.90, decay_episodes = 1500, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.000009, batch_size=256, memory_size=50000):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_episodes = decay_episodes
        self.epsilon_decay = (epsilon - epsilon_min) / decay_episodes
        self.batch_size = batch_size
        self.memory = SequentialMemory(limit=memory_size, window_length=1)
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


    def remember(self, state, action, reward, done):

        self.memory.append(state, action, reward, done)

    def act(self, state, legal_moves):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
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
            return

        # Seleziona un sottoinsieme di transizioni in base ai pesi calcolati
        batch = self.memory.sample(self.batch_size)
        
        states = np.array([transition.state0 for transition in batch])
        actions = np.array([transition.action for transition in batch])
        rewards = np.array([transition.reward for transition in batch])
        terminals = np.array([transition.terminal1 for transition in batch])
        new_states = np.array([transition.state1 for transition in batch])
        new_states = np.squeeze(new_states)

        # Ottieni previsioni in un solo passaggio
        states = states.reshape(self.batch_size, 4, 4)
        new_states = new_states.reshape(self.batch_size, 4, 4)

        targets   = self.model.predict(states, verbose=0)
        new_state_values = np.max(self.model.predict(new_states, verbose=0), axis=1)
        targets[np.arange(self.batch_size), actions] = rewards + (1 - terminals) * self.gamma * new_state_values

        # LOGICA CON TARGET MODEL
        # next_q_values  = self.target_model.predict(new_states, verbose=0)

        # targets[np.arange(self.batch_size), actions] = rewards + (1 - terminals) * self.gamma * np.max(next_q_values, axis=1)

        history = self.model.fit(states, targets, epochs=1, verbose=0)
        #print("Addestro il modello vero e proprio")

        self.loss_history.append(history.history['loss'][0])
        
        self.step_counter += 1
        # if(self.step_counter % 500 == 0):
        #     print("Aggiorno il modello target")
        #     self.update_target_model()
            #print(f"Target model aggiornato al passo {self.step_counter}")

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
