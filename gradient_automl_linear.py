import time
from typing import Dict, Any, Iterable
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import jax
import jax.numpy as jnp
import flax.linen as nn

from losses import double_boundary_loss


def calc_weights_leaky(fin, fout):
    negative_slope = 1e-2
    result = nn.leaky_relu(fin + 1, negative_slope) * nn.leaky_relu(fout, negative_slope)
    return result


def calc_weights_precise(fin, fout):
    result = (fin + 1) * fout
    return result


def total_weights(input_features_size: int, features_arr: np.ndarray) -> int:
    weight_nums = []
    for i_layer in range(len(features_arr)):
        feat_in = input_features_size if i_layer == 0 else features_arr[i_layer - 1]
        feat_out = features_arr[i_layer]

        num_weights = calc_weights_precise(feat_in, feat_out)
        weight_nums.append(num_weights)
    return np.sum(np.array(weight_nums)).item()


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


def discretize_features(features_array):
    step = 16
    features_np = np.array(features_array)
    res_features = []
    for feature in features_np:
        feature_int = int(feature)
        if feature_int <= 0:
            break
        import math
        feature_disc = int(math.ceil(feature_int / step) * step)
        if feature_disc <= 0:
            print("ERROR: feature_disc <= 0")
        res_features.append(feature_disc)
    return np.array(res_features, dtype=np.int32)


def benchmark_features_array(input_features_size, int_features, num_reps=5):

    joint_net = MLP(input_features_size, int_features)
    example = jnp.ones((1000, 10))
    joint_net_vars = joint_net.init(jax.random.PRNGKey(44), example)

    flax_apply_jitted = jax.jit(lambda params, inputs: joint_net.apply(params, inputs))
    _ = flax_apply_jitted(joint_net_vars, example)

    times = []
    for _ in range(num_reps):
        st = time.time()
        _ = flax_apply_jitted(joint_net_vars, example).block_until_ready()
        run_time = time.time() - st
        times.append(run_time)
    
    median_runtime = np.median(np.array(times))
    return median_runtime


def proposal_analytics(input_features_size, features_array, constraints, latency_fn):
    int_features = discretize_features(features_array)

    predicted_lat = latency_fn(int_features).item()

    measured_lat = benchmark_features_array(input_features_size, int_features)

    num_weights = total_weights(input_features_size, int_features)

    print(f"Discretized features: {int_features.tolist()}")
    print(f"predicted {predicted_lat:.6f} measured {measured_lat:.6f} sec, {constraints['latency_sec']}")
    print(f"num_weights {num_weights}, {constraints['parameters']}")

    return dict(predicted_lat=predicted_lat, measured_lat=measured_lat, num_weights=num_weights)


def visualize_results(lat_dicts, seed_lat_dicts, test_lat_dicts, constraints):
    def transpose_list_dict(ld):
        return {k: [d[k] for d in ld] for k in ld[0].keys()}

    opt_dict = transpose_list_dict(lat_dicts)
    seed_dict = transpose_list_dict(seed_lat_dicts)
    test_dict = transpose_list_dict(test_lat_dicts)

    import matplotlib.pyplot as plt
    plt.figure()
    # pred_lats = [v['predicted_lat'] for v in lat_dicts]
    # measured_lats = [v['measured_lat'] for v in lat_dicts]
    # weights_qty = [v['num_weights'] for v in lat_dicts]
    plt.scatter(opt_dict['predicted_lat'], opt_dict['measured_lat'])
    plt.xlabel('pred_lats')
    plt.ylabel('measured_lats')
    plt.savefig('pred_vs_measured.png')

    plt.figure()
    x0 = constraints['parameters']['min']
    x1 = constraints['parameters']['max']
    y0 = constraints['latency_sec']['min']
    y1 = constraints['latency_sec']['max']
    plt.scatter(opt_dict['num_weights'], opt_dict['predicted_lat'], c='blue', label='optimized points')
    plt.scatter(seed_dict['num_weights'], seed_dict['predicted_lat'], c='gray', label='seed points')
    plt.scatter(test_dict['num_weights'], test_dict['predicted_lat'], c='red', label='known solution')
    plt.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], 'g--', label='target region')
    plt.xlabel('num_parameters')
    plt.ylabel('predicted_lats')
    plt.legend()
    plt.savefig('target_region.png')


def gradient_automl_linear(evaluator: Dict[str, Any]):
    min_layers = 5
    max_layers = 20 # 10
    max_latency_sec = 0.002
    min_latency_sec = 0.75 * max_latency_sec
    max_parameters = 1_000_000
    min_parameters = 0.5 * max_parameters
    constraints = dict(
        latency_sec=dict(min=min_latency_sec, max=max_latency_sec),
        parameters=dict(min=min_parameters, max=max_parameters))

    input_features_size = 10

    def predict_latencies(predict_fn, params, features_arr):
        feature_tuples = []
        for i_layer in range(len(features_arr)):
            feat_in = input_features_size if i_layer == 0 else features_arr[i_layer - 1]
            feat_out = features_arr[i_layer]

            features_tuple = (feat_in, feat_out)
            feature_tuples.append(features_tuple)

        features_jnp = jnp.array(feature_tuples)
        raw_latencies = predict_fn.apply({'params': params}, features_jnp)
        return raw_latencies
    
    def rem_loss_fn(features_arr):
        weight_nums = []
        for i_layer in range(len(features_arr)):
            feat_in = input_features_size if i_layer == 0 else features_arr[i_layer - 1]
            feat_out = features_arr[i_layer]

            num_weights = calc_weights_leaky(feat_in, feat_out)
            weight_nums.append(num_weights)

        total_num_weights = jnp.sum(jnp.array(weight_nums))
        num_weights_loss = double_boundary_loss(total_num_weights, min_parameters, max_parameters)

        compactness_array = jnp.maximum(features_arr[1:] - features_arr[:-1], 0)
        compactness_loss = jnp.sum(compactness_array) / jnp.mean(jnp.square(features_arr))

        total_loss = 0.1 * num_weights_loss + 100.0 * compactness_loss

        aux = OrderedDict(
            total_num_weights=total_num_weights, num_weights_loss=num_weights_loss,
            compactness_loss=compactness_loss)

        return total_loss, aux
    
    def total_latency_fn(predict_fn, params, features_arr):
        raw_latencies = predict_latencies(predict_fn, params, features_arr)
        latencies = jnp.maximum(raw_latencies, 0)
        total_latency = jnp.sum(latencies)
        return total_latency

    def latency_loss_fn(predict_fn, params, features_arr):
        total_latency = total_latency_fn(predict_fn, params, features_arr)
        latency_loss = double_boundary_loss(total_latency, min_latency_sec, max_latency_sec)
        aux = OrderedDict(total_latency=total_latency, latency_loss=latency_loss)
        return latency_loss, aux

    predict_fn = evaluator['predict_fn']
    module = evaluator['module']
    params = evaluator['params']
    predict_flax = evaluator['predict_flax']

    from functools import partial
    total_latency_eval_fn = partial(total_latency_fn, predict_flax, params)
    
    test_lat_dicts = []
    for i_pos in range(4, 8):
        test_features_array = jnp.zeros((max_layers,), dtype=jnp.float32) + 512
        test_features_array = test_features_array.at[i_pos].set(-1000)

        print("For", np.array(test_features_array))
        test_lat_dicts.append(proposal_analytics(input_features_size, test_features_array,
            constraints, total_latency_eval_fn))

    gpu = jax.devices("gpu")[0]
    lat_loss, lat_aux_dict = latency_loss_fn(predict_flax, params, test_features_array)
    latency_loss_jit_fn = nn.jit(latency_loss_fn, backend='gpu')
    lat_loss, lat_aux_dict = latency_loss_jit_fn(predict_flax, params, test_features_array)
    lat_grad_jit_fn = nn.jit(jax.value_and_grad(latency_loss_jit_fn, argnums=(2,), has_aux=True), backend='gpu') # , device=gpu
    llg, lat_aux_dict = lat_grad_jit_fn(predict_flax, params, test_features_array)

    raw_latencies = predict_latencies(predict_flax, params, test_features_array)

    tlv = rem_loss_fn(test_features_array)
    
    rem_loss_jit_fn = jax.jit(rem_loss_fn)
    tljv = rem_loss_jit_fn(test_features_array)

    rem_grad_fn = jax.value_and_grad(rem_loss_jit_fn, argnums=(0,), has_aux=True)
    gfnv = rem_grad_fn(test_features_array)

    rem_grad_jit_fn = jax.jit(rem_grad_fn)
    gfnjv = rem_grad_jit_fn(test_features_array)

    print("Optimizing parameters")
    
    from scipy.optimize import minimize
    from scipy.optimize import shgo
    from scipy.stats.qmc import LatinHypercube
    
    def cp_value_and_jacobian(feature_vector: np.ndarray) -> np.ndarray:
        (lv, *_), (lg,) = lat_grad_jit_fn(predict_flax, params, feature_vector)
        (rv, *_), (rg,) = rem_grad_jit_fn(feature_vector)
        total_val = lv + rv
        total_val_np = np.array(total_val)
        total_grad = lg + rg
        total_grad_np = np.array(total_grad)
        return total_val_np.item(), total_grad_np.tolist()

    def print_progress(current_vector):
        print(f"Features {[int(v) for v in current_vector.tolist()]}")
        return False
    
    CH_MAX = 512

    bounds = [(-CH_MAX, CH_MAX) for _ in range(max_layers)]

    fast = False

    num_outer_iters = 1 if fast else 15

    latin_hypercube = LatinHypercube(max_layers)
    seed_points_np = latin_hypercube.random(num_outer_iters) * (CH_MAX - 1) + 1

    results = []
    lat_dicts = []
    seed_lat_dicts = []
    for i_outer in range(num_outer_iters):

        random_features_np = seed_points_np[i_outer]
        print("seed point", random_features_np)

        maxiter = 3 if fast else 30

        res = minimize(
            cp_value_and_jacobian,
            random_features_np,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options=dict(maxiter=maxiter),
            callback=print_progress,
            )

        print(res)
        results.append(res)

        print(">>> random:")
        seed_lat_dict = proposal_analytics(input_features_size, random_features_np,
            constraints, total_latency_eval_fn)
        print("=== optimized:")
        lat_dict = proposal_analytics(input_features_size, res.x,
            constraints, total_latency_eval_fn)
        print("<<<")
        lat_dicts.append(lat_dict)
        seed_lat_dicts.append(seed_lat_dict)

        visualize_results(lat_dicts, seed_lat_dicts, test_lat_dicts, constraints)

    print(results)
    for res in results:
        print("---------------------------------------")
        print_progress(res.x)
    print("---------------------------------------")

    # visualize_results(lat_dicts, seed_lat_dicts, test_lat_dicts, constraints)

    print("Automl done")    
