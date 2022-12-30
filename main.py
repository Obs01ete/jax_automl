import os
import time
import json
from typing import Callable, Dict, Any, Iterable
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
    # dataset_name = f"{op_type}_data.json"
    dataset_name = f"{op_type}_data_1k_.json" # TEMP
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


def calc_weights_leaky(fin, fout):
    # result = jnp.maximum(fin + 1, 0) * jnp.maximum(fout, 0)
    negative_slope = 1e-2
    result = nn.leaky_relu(fin + 1, negative_slope) * nn.leaky_relu(fout, negative_slope)
    return result


def double_boundary_loss(values, min_value, max_value, min_slope=1.0, max_slope=1.0):
    return jnp.maximum(
        max_slope * jnp.maximum(0, (values - max_value) / max_value),
        min_slope * jnp.maximum(0, (min_value - values) / min_value),
        )


class MLP(nn.Module):
    in_feat: int
    hidden_feat: Iterable[int]

    @nn.compact
    def __call__(self, x):
        in_f = self.in_feat
        for out_f in self.hidden_feat:
            x = nn.relu(nn.Dense(in_f, out_f)(x))
            in_f = out_f
        return x


def benchmark_features_array(input_features_size, features_array):
    step = 16
    int_features = (np.ceil(np.array(features_array) / step) * step).astype(np.int32)
    joint_net = MLP(input_features_size, int_features)
    example = jnp.ones((1000, 10))
    joint_net_vars = joint_net.init(jax.random.PRNGKey(44), example)
    # apply_fn = nn.jit(joint_net.apply)
    # pred = apply_fn(joint_net, joint_net_vars, example)

    flax_apply_jitted = jax.jit(lambda params, inputs: joint_net.apply(params, inputs))
    pred = flax_apply_jitted(joint_net_vars, example)

    for _ in range(5):
        st = time.time()
        pred = flax_apply_jitted(joint_net_vars, example).block_until_ready()
        print(time.time() - st)


def gradient_automl(evaluator: Dict[str, Any]):
    min_layers = 5
    max_layers = 10
    max_latency_sec = 0.002
    min_latency_sec = 0.75 * max_latency_sec
    max_parameters = 4_000_000
    min_parameters = 0.5 * max_parameters

    input_features_size = 10
    features_array = jnp.zeros((max_layers,), dtype=jnp.float32) + 300
    # features_array = features_array.at[-2].set(-1000) # TEMP

    benchmark_features_array(input_features_size, features_array)

    class static_ndarray(np.ndarray):
        def __hash__(self):
            return hash(self.tobytes())
    
    def make_static_ndarray(x):
        return static_ndarray(shape=x.shape, dtype=x.dtype, buffer=x.data)

    def predict_latencies(predict_fn, params, features_arr):
        feature_tuples = []
        for i_layer in range(max_layers):
            feat_in = input_features_size if i_layer == 0 else features_arr[i_layer - 1]
            feat_out = features_arr[i_layer]

            features_tuple = (feat_in, feat_out)
            feature_tuples.append(features_tuple)

        features_jnp = jnp.array(feature_tuples)
        raw_latencies = predict_fn.apply({'params': params}, features_jnp)
        return raw_latencies
    
    def rem_loss_fn(features_arr):
        weight_nums = []
        for i_layer in range(max_layers):
            feat_in = input_features_size if i_layer == 0 else features_arr[i_layer - 1]
            feat_out = features_arr[i_layer]

            num_weights = calc_weights_leaky(feat_in, feat_out)
            weight_nums.append(num_weights)

        total_num_weights = jnp.sum(jnp.array(weight_nums))
        num_weights_loss = double_boundary_loss(total_num_weights, min_parameters, max_parameters)

        compactness_array = jnp.maximum(features_arr[1:] - features_arr[:-1], 0)
        compactness_loss = jnp.sum(compactness_array) / jnp.mean(jnp.square(features_arr))
        # compactness_loss = 0.0

        total_loss = 0.1 * num_weights_loss + 100.0 * compactness_loss
        # total_loss = 100.0 * compactness_loss

        aux = OrderedDict(
            total_num_weights=total_num_weights, num_weights_loss=num_weights_loss,
            compactness_loss=compactness_loss)

        return total_loss, aux
    
    def latency_loss_fn(predict_fn, params, features_arr):
        raw_latencies = predict_latencies(predict_fn, params, features_arr)
        latencies = jnp.maximum(raw_latencies, 0)
        total_latency = jnp.sum(latencies)
        latency_loss = double_boundary_loss(total_latency, min_latency_sec, max_latency_sec)
        aux = OrderedDict(total_latency=total_latency, latency_loss=latency_loss,
            raw_latencies=raw_latencies)
        return latency_loss, aux

    predict_fn = evaluator['predict_fn']
    module = evaluator['module']
    params = evaluator['params']
    predict_flax = evaluator['predict_flax']

    # static_params = jax.tree_util.tree_map(make_static_ndarray, params)
    # static_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)

    gpu = jax.devices("gpu")[0]
    lat_loss, lat_aux_dict = latency_loss_fn(predict_flax, params, features_array)
    latency_loss_jit_fn = nn.jit(latency_loss_fn, backend='gpu') # , device=gpu
    lat_loss, lat_aux_dict = latency_loss_jit_fn(predict_flax, params, features_array)
    lat_grad_fn = nn.jit(jax.value_and_grad(latency_loss_jit_fn, argnums=(2,), has_aux=True), backend='gpu') # , device=gpu
    llg, lat_aux_dict = lat_grad_fn(predict_flax, params, features_array)

    raw_latencies = predict_latencies(predict_flax, params, features_array)
    # predict_latencies_grad = jax.grad(predict_latencies, argnums=(2,))
    # plg = predict_latencies_grad(predict_flax, params, features_array)

    tlv = rem_loss_fn(features_array)
    
    rem_loss_jit_fn = jax.jit(rem_loss_fn) # static_argnums=(1, 2)
    tljv = rem_loss_jit_fn(features_array)

    rem_grad_fn = jax.value_and_grad(rem_loss_jit_fn, argnums=(0,), has_aux=True)
    gfnv = rem_grad_fn(features_array)

    rem_grad_fn_jit = jax.jit(rem_grad_fn)
    gfnjv = rem_grad_fn_jit(features_array)

    # from jax.test_util import check_grads
    # check_grads(rem_loss, (features_array,), order=1)

    print("Optimizing parameters")

    for i_step in tqdm(range(100)):
        st = time.time()
        (lat_loss, lat_aux_dict), (lat_grad,) = lat_grad_fn(predict_flax, params, features_array)
        lat_loss.block_until_ready()
        print(time.time() - st)
        st = time.time()
        (rem_loss, rem_aux_dict), (rem_grad,) = rem_grad_fn_jit(features_array)
        rem_loss.block_until_ready()
        print(time.time() - st)
        total_grad = lat_grad + rem_grad
        grad_scaled_mimimize = - 5e+4 * np.array(total_grad)
        print(f"--- step={i_step} lat_loss={lat_loss.item():.6f} rem_loss={rem_loss.item():.6f}")
        lat_aux_dict = {k: v.item() if v.shape == () else v for k, v in lat_aux_dict.items()}
        rem_aux_dict = {k: v.item() if v.shape == () else v for k, v in rem_aux_dict.items()}
        print(f"lat_aux_dict={lat_aux_dict}")
        print(f"rem_aux_dict={rem_aux_dict}")
        print(f"grad={grad_scaled_mimimize}")
        if jnp.mean(jnp.abs(grad_scaled_mimimize)).item() < 1e-6:
            # print("Grads are zero. Early stopping.")
            # break
            pass
        features_array += grad_scaled_mimimize
        print(f"features_array={features_array.astype(np.int32)}")
        benchmark_features_array(input_features_size, features_array)

    print(features_array)

    benchmark_features_array(input_features_size, features_array)

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
