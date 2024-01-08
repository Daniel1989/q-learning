from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
import numpy as np
import time
class Agent:
    def __init__(self, grid_size, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01, gamma=0.99):
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.model = self.build_model()
        self.temp1 = None
        self.temp2 = None

    def build_model(self):
        # Create a sequential model with 3 layers
        model = Sequential([
            # Input layer expects a flattened grid, hence the input shape is grid_size squared
            # input shape最后都会包一个batch，默认是1，所以这里说明input_shape (25,)这里就一维，但是
            # 由于最外面默认要包一层，一个batch，所以输入是一个二维数组，然后里面的就一个item，长度是25的数组
            # 可以通过np.reshape来满足
            Dense(128, activation='relu', input_shape=(self.grid_size ** 2,)),
            Dense(64, activation='relu'),
            # Output layer with 4 units for the possible actions (up, down, left, right)
            # 同理，输出也会默认包一层，就是代表一个batch
            Dense(4, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')

        return model

    def get_action(self, state):

        # rand() returns a random value between 0 and 1
        if np.random.rand() <= self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, 4)
        else:
            # Add an extra dimension to the state to create a batch with one instance
            print(f"old state is {state.shape}")
            state = np.expand_dims(state, axis=0)
            print(f"new state is {state.shape}")
            # Use the model to predict the Q-values (action values) for the given state
            q_values = self.model.predict(state, verbose=0)
            print(f"q_values is {q_values}")
            # Select and return the action with the highest Q-value
            action = np.argmax(q_values[0])  # Take the action from the first (and only) entry
            print(f"action is {action}")
        # Decay the epsilon value to reduce the exploration over time
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return action

    def learn(self, experiences, episode, step):
        states = np.array([experience.state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        dones = np.array([experience.done for experience in experiences])

        # Predict the Q-values (action values) for the given state batch
        current_q_values = self.model.predict(states, verbose=0)
        print(f"current_q_values is {current_q_values}")
        # Predict the Q-values for the next_state batch
        next_q_values = self.model.predict(next_states, verbose=0)

        # Initialize the target Q-values as the current Q-values
        target_q_values = current_q_values.copy()

        # Loop through each experience in the batch
        for i in range(len(experiences)):
            if dones[i]:
                # If the episode is done, there is no next Q-value
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # The updated Q-value is the reward plus the discounted max Q-value for the next state
                # [i, actions[i]] is the numpy equivalent of [i][actions[i]]
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        print(f"episode {episode}, step {step}")
        # 这里长度都是 32，然后里面的维度是 4,32是batch的大小
        print(f"states length is {len(states)}, target_q_values length is {len(target_q_values)}")
        start_time = time.time()
        if self.temp1 is None:
            self.temp1 = states[0]
        if self.temp2 is None:
            self.temp2 = target_q_values[0]
        print(f"fit states is {self.temp1}")
        print(f"fit target_q_values is {self.temp2}")
        # 序列模型常用于图片分类，相似的，就是一个图有很多维数组，最后变成一个一维四个元素数组。本质上就是映射到一个四个分类的过程，看哪个分类值比较大
        tf_callback = callbacks.TensorBoard(log_dir="./logs")
        self.model.fit(states, target_q_values, epochs=1, verbose=0, callbacks=[tf_callback])
        end_time = time.time()
        print(f"train cost time is: {end_time - start_time}")

    def load(self, file_path):
        self.model = load_model(file_path)

    def save(self, file_path):
        self.model.save(file_path)