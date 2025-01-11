import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras import regularizers
from collections import deque
import random


class AdaptiveLearningRateScheduler:
    def __init__(self, initial_rate=0.001, min_rate=1e-5, max_rate=0.01, factor_increase=1.2, factor_decrease=0.5, patience=5):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.factor_increase = factor_increase
        self.factor_decrease = factor_decrease
        self.patience = patience
        self.wait = 0
        self.best_score = -float('inf')

    def update(self, current_score):
        if current_score > self.best_score:
            # Miglioramento dello score: potremmo aumentare il learning rate
            self.best_score = current_score
            self.wait = 0
            self.current_rate = min(self.current_rate * self.factor_increase, self.max_rate)
        else:
            # Nessun miglioramento: aspetta e riduce se necessario
            self.wait += 1
            if self.wait >= self.patience:
                self.current_rate = max(self.current_rate * self.factor_decrease, self.min_rate)
                self.wait = 0
        return self.current_rate


class AdaptiveEpsilon:
    def __init__(self, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.995, patience=50):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.patience = patience
        self.best_score = -float('inf')
        self.wait = 0

    def update(self, current_score):
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Rallenta il decadimento di epsilon
                self.decay_rate = min(self.decay_rate * 1.1, 0.999)
                self.wait = 0
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)
        return self.epsilon
    
class DQNModel:
    def __init__(self, state_shape, action_space, learning_rate=0.001):
        self.state_shape = state_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.model = self._build_model()


    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.1))  # Dropout rate of 20%
        
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.1))
        
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.1))

        model.add(BatchNormalization())

        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.1))  # Dropout rate of 20%
        
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.1))
        
        model.add(Dense(8, activation='relu', kernel_initializer='he_uniform',kernel_regularizer=regularizers.l2(1e-5)))
        model.add(Dropout(0.1))

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
        self.model = DQNModel(state_shape, action_space).model
        self.target_model = DQNModel(state_shape, action_space).model
        self.update_target_model()
        self.step_counter = 0
        self.loss_history = []
        self.adaptive_epsilon = AdaptiveEpsilon(initial_epsilon=1, min_epsilon=0.01, decay_rate=0.995, patience=20)
        self.lr_scheduler = AdaptiveLearningRateScheduler(initial_rate=0.001, patience=10)
        self.current_lr = 0



    def update_target_model(self, tau=0.005):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = [
            tau * mw + (1 - tau) * tw
            for mw, tw in zip(model_weights, target_weights)
        ]
        self.target_model.set_weights(new_weights)



    def remember(self, transition):
        #print("Sto salvando in memory gli stati")
        self.memory.append((transition))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def process_episode(self, episode_memory, final_score):
        self.batch_size = len(episode_memory)
        # Estrai dati dall'episodio
        states = np.array([transition[0] for transition in episode_memory])
        next_states = np.array([transition[3] for transition in episode_memory])
        actions = [transition[1] for transition in episode_memory]
        rewards = [transition[2] for transition in episode_memory]
        dones = [transition[4] for transition in episode_memory]

        # Predizioni batch
        q_values_current = self.model.predict(states, verbose=0)
        q_values_next_model = self.model.predict(next_states, verbose=0)
        q_values_next_target = self.target_model.predict(next_states, verbose=0)

        # Aggiorna i Q-values
        for i in range(len(episode_memory)):
            if dones[i]:
                target = rewards[i]
            else:
                best_action = np.argmax(q_values_next_model[i])
                target = rewards[i] + self.gamma * q_values_next_target[i][best_action]
            q_values_current[i][actions[i]] = target

        if len(self.memory) >= self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            past_states = np.array([transition[0] for transition in batch])
            past_next_states = np.array([transition[3] for transition in batch])
            past_actions = [transition[1] for transition in batch]
            past_rewards = [transition[2] for transition in batch]
            past_dones = [transition[4] for transition in batch]

            # Predizioni batch per il buffer di replay
            q_values_past = self.model.predict(past_states, verbose=0)
            q_values_past_next_model = self.model.predict(past_next_states, verbose=0)
            q_values_past_next_target = self.target_model.predict(past_next_states, verbose=0)

            # Aggiorna i Q-values per le esperienze passate
            for i in range(self.batch_size):
                if past_dones[i]:
                    target = past_rewards[i]
                else:
                    past_best_action = np.argmax(q_values_past_next_model[i])
                    target = past_rewards[i] + self.gamma * q_values_past_next_target[i][past_best_action]
                q_values_past[i][past_actions[i]] = target

            # Combina i dati dell'episodio corrente con quelli passati
            combined_states = np.concatenate([states, past_states])
            combined_q_values = np.concatenate([q_values_current, q_values_past])
        else:
            # Se il buffer non è sufficiente, usa solo l'episodio corrente
            combined_states = states
            combined_q_values = q_values_current

        #print(f"Numero del buffer finale: {len(combined_states)}")
        # Addestra il modello
        self.epsilon = self.adaptive_epsilon.update(final_score)
        self.current_lr = self.lr_scheduler.update(final_score)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.current_lr)
        history = self.model.fit(combined_states, combined_q_values, batch_size=200, verbose=0)
        self.loss_history.append(history.history['loss'][0])

        # Aggiungi al replay buffer
        for transition in episode_memory:
            self.remember(transition)

        
        self.step_counter += 1
        if(self.step_counter % 10 == 0):
            self.update_target_model()
            #print(f"Target model aggiornato al passo {self.step_counter}")

        if self.step_counter % 1000 == 0 or len(self.memory) >= 50000:
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
