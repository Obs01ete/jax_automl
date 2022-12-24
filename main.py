import os
import time
import json
from typing import Callable
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from benchmarking import create_dataset
from latency_model import LatencyModelTrainer, LatencyEvaluator


def load_or_create_dataset():
    # cpus = jax.devices("cpu")
    gpus = jax.devices("gpu")
    gpu = gpus[0]

    op_type = 'linear' # 'linear' 'conv2d'
    dataset_name = f"{op_type}_data.json"
    if os.path.exists(dataset_name):
        with open(dataset_name, "r") as f:
            dataset = json.load(f)
    else:
        num_samples = 20000 # 1000
        print(f"Measuring {num_samples} samples")
        time_start = time.time()
        dataset = create_dataset(gpu, op_type, num_samples)
        gen_wall_time = time.time() - time_start
        print(f"Generated in {int(gen_wall_time)} seconds")
        with open(dataset_name, "w") as f:
            json.dump(dataset, f, indent=4)

    return dataset


def dataset_analytics(dataset):
    latencies = [r['target'] for r in dataset['dataset']]
    features = np.array([r['features'] for r in dataset['dataset']])

    # plt.figure()
    # plt.hist(latencies, bins=200)
    # plt.yscale('log')
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.hist(features[:, 0], bins=200)
    # plt.yscale('log')
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.hist(features[:, 1], bins=200)
    # plt.yscale('log')
    # plt.grid()
    # plt.show()

    plt.figure()
    plt.scatter(features[:, 0]*features[:, 1], latencies, marker='.')
    plt.grid()
    plt.show()


def gradient_automl(evaluator: LatencyEvaluator):
    min_layers = 5
    max_layers = 10
    max_latency_sec = 0.005
    min_latency_sec = 0.9 * max_latency_sec
    max_parameters = 1_000_000
    min_parameters = max_parameters // 2

    input_features_size = 10
    features_array = np.zeros((max_layers,), dtype=np.float32) + 100

    # @jax.jit
    def total_loss(features_arr, evaluator_latency_fn: Callable, evaluator_params):
        latencies = []
        for i_layer in range(max_layers):
            feat_in = input_features_size if i_layer == 0 else features_arr[i_layer - 1]
            feat_out = features_arr[i_layer]
            features_jnp = jnp.expand_dims(jnp.array((feat_in, feat_out)), axis=0)
            latency = evaluator_latency_fn(evaluator_params, features_jnp)
            latency = jnp.max(jnp.array([latency, jnp.zeros_like(latency)]), axis=0)
            latencies.append(latency)
        total_latency = jnp.sum(jnp.array(latencies))
        latency_loss = jnp.max(jnp.array([
            jnp.max(jnp.array([0, (total_latency - max_latency_sec) / max_latency_sec])),
            jnp.max(jnp.array([0, (min_latency_sec - total_latency) / min_latency_sec])),
            ]))
        total_loss = latency_loss
        return total_loss, total_latency

    grad_fn = jax.value_and_grad(total_loss, argnums=(0,), has_aux=True)

    def grad_fn_partial(features_arr):
        return grad_fn(features_arr, evaluator.latency_fn(), evaluator.params()) 

    print("Optimizing parameters")

    for i_step in tqdm(range(100)):
        (loss, latency), (grad,) = grad_fn_partial(features_array)
        grad_scaled_mimimize = - 5e+5 * np.array(grad)
        print(f"step={i_step} loss={loss.item():.6f}, latency={latency.item():.6f}")
        print(f"grad={grad_scaled_mimimize}")
        if jnp.mean(jnp.abs(grad_scaled_mimimize)).item() < 1e-6:
            print("Grads are zero. Early stopping.")
            break
        features_array += grad_scaled_mimimize
        print(f"features_array={features_array}")

    print(features_array)

    print("Automl done")    


def main():

    dataset = load_or_create_dataset()

    if False:
        dataset_analytics(dataset)

    trainer = LatencyModelTrainer(dataset)
    trainer.load_or_train()
    # trainer.evaluate()

    gradient_automl(trainer.get_evaluator())

    print("Done")


if __name__ == "__main__":
    main()
