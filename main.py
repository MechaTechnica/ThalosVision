import minst_loader as minst_loader
import NeuralNetMLP as nn
import matplotlib.pyplot as plt

x_train, y_train = minst_loader.load_minst('/Users/Rob/Documents/NeuralNetworks/mnist/', kind='train')
x_test, y_test = minst_loader.load_minst('/Users/Rob/Documents/NeuralNetworks/mnist/', kind='t10k')

nn = nn.NeuralNetMLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)
nn.fit(x_train=x_train[:55000], y_train=y_train[:55000], x_valid=x_train[55000:], y_valid=y_train[55000:])

plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()