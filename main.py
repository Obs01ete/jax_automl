import os
import time
import json
from typing import Callable, Dict, Any
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

import jax
import jax.numpy as jnp
import flax.linen as nn

from benchmarking import create_dataset
from latency_model import LatencyModelTrainer, LatencyNet


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


def calc_weights(fin, fout):
    return jnp.maximum(fin + 1, 0) * jnp.maximum(fout, 0)


def double_boundary_loss(values, min_value, max_value, min_slope=1.0, max_slope=1.0):
    return jnp.maximum(
        max_slope * jnp.maximum(0, (values - max_value) / max_value),
        min_slope * jnp.maximum(0, (min_value - values) / min_value),
        )

import flax

def gradient_automl(evaluator: Dict[str, Any]):
    min_layers = 5
    max_layers = 10
    max_latency_sec = 0.001
    min_latency_sec = 0.75 * max_latency_sec
    max_parameters = 4_000_000
    min_parameters = 0.5 * max_parameters

    input_features_size = 10
    features_array = jnp.zeros((max_layers,), dtype=jnp.float32) + 300
    features_array = features_array.at[-2].set(-1000) # TEMP

    class static_ndarray(np.ndarray):
        def __hash__(self):
            return hash(self.tobytes())
    
    def make_static_ndarray(x):
        return static_ndarray(shape=x.shape, dtype=x.dtype, buffer=x.data)

    def total_loss(features_arr, module, params):
        latencies = []
        weight_nums = []
        for i_layer in range(max_layers):
            feat_in = input_features_size if i_layer == 0 else features_arr[i_layer - 1]
            feat_out = features_arr[i_layer]

            features_tuple = (feat_in, feat_out)
            features_np = jnp.expand_dims(jnp.array(features_tuple, dtype=jnp.float32), axis=0)
            # features_static = make_static_ndarray(np.array(features_tuple))
            latency = module.apply({'params': params}, features_np)
            latency = latency[0]
            latency = jnp.maximum(latency, 0)
            latencies.append(latency)

            num_weights = calc_weights(feat_in, feat_out)
            weight_nums.append(num_weights)

        total_latency = jnp.sum(jnp.array(latencies))
        latency_loss = double_boundary_loss(total_latency, min_latency_sec, max_latency_sec)

        total_num_weights = jnp.sum(jnp.array(weight_nums))
        num_weights_loss = double_boundary_loss(total_num_weights, min_parameters, max_parameters)

        # compactness_array = jnp.maximum(
        #     jnp.maximum(features_array[1:], 0),
        #     -jnp.minimum(features_array[:-1], 0)
        # )
        # compactness_array = jnp.sqrt(
        #     jax.lax.clamp(0.0, features_array[1:], 1e+6) *
        #     (-jax.lax.clamp(-1e+6, features_array[:-1], 0.0))
        # )
        compactness_array = \
            jax.nn.softplus(features_array[1:]) * \
            jax.nn.softplus(-features_array[:-1])
        compactness_loss = jnp.sum(compactness_array) / jnp.sum(jnp.square(features_array))

        total_loss = latency_loss + 0.1 * num_weights_loss + 10.0 * compactness_loss
        # total_loss = 10.0 * compactness_loss

        # print(f"latency_loss={latency_loss.item()} num_weights_loss={num_weights_loss.item()}")

        aux = OrderedDict(total_latency=total_latency, latency_loss=latency_loss,
            total_num_weights=total_num_weights, num_weights_loss=num_weights_loss,
            compactness_loss=compactness_loss)
        # aux = OrderedDict(compactness_loss=compactness_loss)

        return total_loss, aux
    
    module = evaluator['module']
    params = evaluator['params']

    # dense_class = nn.Dense
    # out_feat = 700
    # x = np.ones((100, 600))

    # dense_inst = dense_class(out_feat, name=f'layers_7')
    # dense_vars = dense_inst.init(jax.random.PRNGKey(0), x)
    # dense_inst = dense_inst.bind(dense_vars)
    # y = dense_inst(x).block_until_ready()
    # st = time.time()
    # for _ in range(10):
    #     y = dense_inst(x).block_until_ready()
    # print(time.time() - st)

    # jitted_call = jax.jit(lambda params, inputs: dense_inst.apply(params, inputs))
    # y = jitted_call(dense_vars, x)
    # st = time.time()
    # for _ in range(10):
    #     y = jitted_call(dense_vars, x).block_until_ready()
    # print(time.time() - st)
    
    static_params = jax.tree_util.tree_map(make_static_ndarray, params)
    # static_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    
    tlv = total_loss(features_array, module, static_params)
    
    total_loss_jit = jax.jit(total_loss, static_argnums=(1, 2)) # static_argnums=(1, 2)
    tljv = total_loss_jit(features_array, module, static_params)

    grad_fn = jax.jit(jax.value_and_grad(total_loss_jit, argnums=(0,), has_aux=True))
    gfnv = grad_fn(features_array, module, static_params)

    grad_fn_jit = jax.jit(grad_fn)
    gfnjv = grad_fn_jit(features_array, module, static_params)

    # def grad_fn_partial(features_arr):
    #     return grad_fn(features_arr, evaluator['apply_fn'], evaluator['params']) 

    print("Optimizing parameters")

    for i_step in tqdm(range(100)):
        (loss, aux_dict), (grad,) = grad_fn_jit(features_array, module, static_params)
        grad_scaled_mimimize = - 5e+15 * np.array(grad)
        print(f"step={i_step} loss={loss.item():.6f}")
        aux_dict = {k: v.item() for k, v in aux_dict.items()}
        print(f"aux_dict={aux_dict}")
        print(f"grad={grad_scaled_mimimize}")
        if jnp.mean(jnp.abs(grad_scaled_mimimize)).item() < 1e-6:
            # print("Grads are zero. Early stopping.")
            # break
            pass
        features_array += grad_scaled_mimimize
        print(f"features_array={features_array.astype(np.int32)}")

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
