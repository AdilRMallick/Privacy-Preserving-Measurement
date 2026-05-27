import numpy as np
import tensorflow as tf
from typing import List, Tuple


def load_and_partition_mnist(
    num_clients: int,
    iid: bool = True,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """Load MNIST and partition training data across clients."""
    rng = np.random.default_rng(seed)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    client_data: List[Tuple[np.ndarray, np.ndarray]] = []

    if iid:
        shuffled = rng.permutation(len(x_train))
        for shard in np.array_split(shuffled, num_clients):
            client_data.append((x_train[shard], y_train[shard]))
    else:
        # Non-IID: sort by label, assign 2 label-contiguous shards per client
        sorted_idx = np.argsort(y_train)
        num_shards = num_clients * 2
        shards = np.array_split(sorted_idx, num_shards)
        shard_order = rng.permutation(num_shards)
        for i in range(num_clients):
            idx = np.concatenate([shards[shard_order[2 * i]], shards[shard_order[2 * i + 1]]])
            client_data.append((x_train[idx], y_train[idx]))

    return client_data, (x_test, y_test)
