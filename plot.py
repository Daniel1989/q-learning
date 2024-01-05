import matplotlib.pyplot as plt


def plot_history(history):

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Step')
    plt.plot(history['epoch'], history['with_model_step'],
             label='model Steps')
    plt.plot(history['epoch'], history['no_model_step'],
             label='no model Steps')
    plt.legend()

    plt.show()
