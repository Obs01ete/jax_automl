import os
import time
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from benchmarking import create_dataset
from latency_model import LatencyModelTrainer


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


def main():

    dataset = load_or_create_dataset()

    if False:
        dataset_analytics(dataset)

    trainer = LatencyModelTrainer(dataset)
    trainer.load_or_train()
    trainer.evaluate()

    print("Done")


if __name__ == "__main__":
    main()
