from qlearn.Environment import Environment
from qlearn.Agent import Agent
from qlearn.ExperienceReplay import ExperienceReplay
from plot import plot_history
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
history = {
    'epoch': [],
    'with_model_step': [],
    'no_model_step': []
}


def start(enable_model: bool = False):
    grid_size = 5

    environment = Environment(grid_size=grid_size, render_on=True)
    agent = Agent(grid_size=grid_size, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01)
    if enable_model:
        agent.load(f'models/model_{grid_size}.h5')

    experience_replay = ExperienceReplay(capacity=10000, batch_size=32)

    # Number of episodes to run before training stops
    episodes = 100
    # Max number of steps in each episode
    max_steps = 500

    for episode in range(episodes):

        # Get the initial state of the environment and set done to False
        state = environment.reset()

        # Loop until the episode finishes
        for step in range(max_steps):
            # print('Episode:', episode)
            # print('Step:', step)
            # print('Epsilon:', agent.epsilon)

            # Get the action choice from the agents policy
            action = agent.get_action(state)

            # Take a step in the environment and save the experience
            reward, next_state, done = environment.step(action)
            experience_replay.add_experience(state, action, reward, next_state, done)

            # If the experience replay has enough memory to provide a sample, train the agent
            if experience_replay.can_provide_sample():
                experiences = experience_replay.sample_batch()
                agent.learn(experiences, episode, step)

            # Set the state to the next_state
            state = next_state

            if done:
                print('finish Episode:', episode)
                print('finish Step:', step)
                if enable_model:
                    history['with_model_step'].append(step)
                else:
                    history['epoch'].append(episode)
                    history['no_model_step'].append(step)
                break

            # Optionally, pause for half a second to evaluate the model
            # time.sleep(0.5)

        agent.save(f'models/model_{grid_size}.h5')


if __name__ == '__main__':
    start()
    # start(True)
    # print(
    #     f"with_model_step length is: {len(history['with_model_step'])}, average step is {sum(history['with_model_step']) / len(history['with_model_step'])}")
    #
    # print(
    #     f"no_model_step length is: {len(history['no_model_step'])}, average step is {sum(history['no_model_step']) / len(history['no_model_step'])}")
    #
    # plot_history(history)
