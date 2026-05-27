import random
from typing import Any, Dict

import numpy as np

from differential_privacy.accountant import RDPAccountant
from federated.client import FederatedClient
from federated.data_utils import load_and_partition_mnist
from federated.model import build_mnist_model
from federated.server import FederatedServer


def run_federated_experiment(
    fed_config: Dict[str, Any],
    dp_config: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run one complete DP-FedAvg experiment on MNIST.

    Returns a dict with per-checkpoint history and final accuracy / epsilon.
    """
    np.random.seed(fed_config["seed"])
    random.seed(fed_config["seed"])

    client_data, (x_test, y_test) = load_and_partition_mnist(
        fed_config["num_clients"],
        iid=fed_config["iid"],
        seed=fed_config["seed"],
    )

    lr = fed_config["learning_rate"]
    model_fn = lambda: build_mnist_model(lr)

    client_cfg = {
        "local_epochs": fed_config["local_epochs"],
        "batch_size": fed_config["batch_size"],
        "clip_norm": fed_config["clip_norm"],
    }
    clients = [
        FederatedClient(i, client_data[i], model_fn, client_cfg)
        for i in range(fed_config["num_clients"])
    ]

    server_dp = (
        {
            "add_noise": True,
            "noise_multiplier": dp_config["noise_multiplier"],
            "clip_norm": fed_config["clip_norm"],
        }
        if dp_config.get("add_noise")
        else {}
    )
    server = FederatedServer(model_fn, server_dp)

    accountant = (
        RDPAccountant(
            noise_multiplier=dp_config["noise_multiplier"],
            sampling_rate=fed_config["clients_per_round"] / fed_config["num_clients"],
            delta=dp_config["delta"],
        )
        if dp_config.get("add_noise")
        else None
    )

    num_rounds = fed_config["num_rounds"]
    log_every = max(1, num_rounds // 10)
    history: Dict[str, list] = {"round": [], "accuracy": [], "loss": [], "epsilon": []}

    for r in range(1, num_rounds + 1):
        selected = random.sample(range(fed_config["num_clients"]), fed_config["clients_per_round"])
        global_w = server.get_weights()

        updates, counts = [], []
        for cid in selected:
            delta, n = clients[cid].compute_update(global_w)
            updates.append(delta)
            counts.append(n)

        server.aggregate(updates, counts)

        if accountant:
            eps = accountant.accumulate(1)
        else:
            eps = float("inf")

        if r % log_every == 0 or r == num_rounds:
            loss, acc = server.evaluate(x_test, y_test)
            history["round"].append(r)
            history["accuracy"].append(acc)
            history["loss"].append(loss)
            history["epsilon"].append(eps)

            if verbose:
                eps_str = f"{eps:7.3f}" if eps < 1e6 else "      ∞"
                delta_str = f"{dp_config.get('delta', 0):.0e}"
                print(
                    f"  Round {r:3d}/{num_rounds} | "
                    f"Acc: {acc:.4f} | Loss: {loss:.4f} | "
                    f"ε = {eps_str}  (δ = {delta_str})"
                )

    return {
        "history": history,
        "final_accuracy": history["accuracy"][-1],
        "final_loss": history["loss"][-1],
        "final_epsilon": history["epsilon"][-1],
    }
