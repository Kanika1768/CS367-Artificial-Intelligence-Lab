import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), _ = mnist.load_data()
X = (x_train[:12000].reshape(-1, 784) > 80).astype(np.float32)

rng = np.random.default_rng(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def sample(probability):
    return (rng.random(probability.shape) < probability).astype(np.float32)


visible, hidden = 784, 128
W = rng.normal(0, 0.01, (visible, hidden)).astype(np.float32)

pixel_mean = X.mean(axis=0)
visible_bias = np.log((pixel_mean + 0.01) / (1 - pixel_mean + 0.01))
hidden_bias = np.zeros(hidden, dtype=np.float32)

epochs, batch_size, learning_rate = 25, 100, 0.05
losses = []

print("Training Restricted Boltzmann Machine...")
for epoch in range(epochs):
    order = rng.permutation(len(X))
    total_error = 0

    for start in range(0, len(X), batch_size):
        v0 = X[order[start:start + batch_size]]

        h0_prob = sigmoid(v0 @ W + hidden_bias)
        h0 = sample(h0_prob)

        v1_prob = sigmoid(h0 @ W.T + visible_bias)
        v1 = sample(v1_prob)
        h1_prob = sigmoid(v1 @ W + hidden_bias)

        n = len(v0)
        W += learning_rate * (v0.T @ h0_prob - v1.T @ h1_prob) / n
        visible_bias += learning_rate * np.mean(v0 - v1, axis=0)
        hidden_bias += learning_rate * np.mean(h0_prob - h1_prob, axis=0)

        total_error += np.mean((v0 - v1_prob) ** 2) * n

    loss = total_error / len(X)
    losses.append(loss)
    print(f"Epoch {epoch + 1:2}/{epochs}: reconstruction MSE = {loss:.4f}")

chosen = [np.where(y_train[12000:] == d)[0][0] + 12000 for d in range(10)]
original = x_train[chosen]
v_test = (original.reshape(-1, 784) > 80).astype(np.float32)
h_test = sample(sigmoid(v_test @ W + hidden_bias))
reconstructed = sigmoid(h_test @ W.T + visible_bias)

v = sample(np.broadcast_to(pixel_mean, (10, visible)))
for _ in range(500):
    h = sample(sigmoid(v @ W + hidden_bias))
    generated_prob = sigmoid(h @ W.T + visible_bias)
    v = sample(generated_prob)

fig, axes = plt.subplots(4, 10, figsize=(14, 6))
rows = [original, reconstructed, generated_prob, v]
labels = ["Original", "Reconstructed", "Generated (soft)", "Generated (binary)"]
for row, (images, label) in enumerate(zip(rows, labels)):
    for col in range(10):
        axes[row, col].imshow(images[col].reshape(28, 28), cmap="gray", vmin=0, vmax=255 if row == 0 else 1)
        axes[row, col].axis("off")
fig.suptitle("MNIST: Real, Reconstructed, and RBM Generated Digits")
fig.tight_layout(rect=[0.09, 0, 1, 0.96])
for row, label in enumerate(labels):
    box = axes[row, 0].get_position()
    fig.text(0.025, (box.y0 + box.y1) / 2, label, rotation=90,
             va="center", ha="center", fontsize=10)
fig.savefig("rbm_results.png", dpi=170, bbox_inches="tight")

plt.show()
