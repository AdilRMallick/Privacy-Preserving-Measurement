FEDERATED = {
    "num_clients": 100,
    "clients_per_round": 10,
    "num_rounds": 50,
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.01,
    "clip_norm": 1.0,
    "iid": True,
    "seed": 42,
}

DP = {
    "add_noise": True,
    "noise_multiplier": 1.1,
    "delta": 1e-5,
}

TRADEOFF = {
    "num_rounds": 20,
    "noise_multipliers": [0.5, 1.0, 1.5, 2.0],
}
