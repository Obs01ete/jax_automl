import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

import jax

from benchmarking import create_dataset
from latency_model import LatencyModelTrainer
from gradient_automl_linear import gradient_automl_linear
from gradient_automl_conv2d import gradient_automl_conv2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op-type', default='conv2d', type=str)
    parser.add_argument('--num-samples', default=20000, type=int)
    args = parser.parse_args()

    op_type = args.op_type
    num_samples = args.num_samples

    dataset = load_or_create_dataset(op_type, num_samples)

    if False:
        dataset_analytics(dataset)

    trainer = LatencyModelTrainer(dataset, op_type)
    trainer.load_or_train()
    # trainer.evaluate()

    evaluator = trainer.get_evaluator()

    if op_type == 'linear':
        gradient_automl_linear(evaluator)
    else:
        gradient_automl_conv2d(evaluator)

    print("Done")


def load_or_create_dataset(
    op_type='linear',
    num_samples=1000,
):
    assert op_type in ('linear', 'conv2d')

    # cpus = jax.devices("cpu")
    gpus = jax.devices("gpu")
    gpu = gpus[0]

    dataset_name = f"{op_type}_data.json"
    # dataset_name = f"{op_type}_data_11k.json" # TEMP

    if os.path.exists(dataset_name):
        with open(dataset_name, "r") as f:
            dataset = json.load(f)
    else:
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
    feature_names = dataset['feature_names']

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
    if len(feature_names) == 2:
        plt.scatter(features[:, 0]*features[:, 1], latencies, marker='.')
    else:
        flops = np.product(features[:, :7], axis=-1) / \
            np.product(features[:, 7:], axis=-1)
        plt.scatter(flops, latencies, marker='.')
    plt.xlabel("FLOPs")
    plt.ylabel("latency sec")
    plt.grid()
    plt.show()

    print("")


if __name__ == "__main__":
    main()
