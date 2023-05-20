import time
from typing import Dict, Any, Iterable, Tuple, List
import numpy as np
from collections import OrderedDict
from functools import partial
from multiprocessing import get_context

import jax
import jax.numpy as jnp
import flax.linen as nn

from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from losses import double_boundary_loss
from constraints import Constraints, MinMax
from visualization import visualize_results
from benchmarking import BENCHMARKING_BATCH


def calc_weights_leaky(fin, fout):
    negative_slope = 1e-2
    result = nn.leaky_relu(fin + 1, negative_slope) * \
        nn.leaky_relu(fout, negative_slope)
    return result


def calc_weights_precise(fin, fout):
    result = (fin + 1) * fout
    return result


def total_weights(input_features_size: int, features_arr: np.ndarray) -> int:
    weight_nums = []
    for i_layer in range(len(features_arr)):
        feat_in = input_features_size if i_layer == 0 \
            else features_arr[i_layer - 1]
        feat_out = features_arr[i_layer]

        num_weights = calc_weights_precise(feat_in, feat_out)
        weight_nums.append(num_weights)
    return np.sum(np.array(weight_nums)).item()


class MLP(nn.Module):
    in_feat: int
    hidden_feat: Iterable[int]

    @nn.compact
    def __call__(self, x):
        for out_f in self.hidden_feat:
            x = nn.relu(nn.Dense(out_f)(x))  # bug fixed
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
    example = jnp.ones((BENCHMARKING_BATCH, input_features_size))
    joint_net_vars = joint_net.init(jax.random.PRNGKey(44), example)

    flax_apply_jitted = jax.jit(lambda params,
                                inputs: joint_net.apply(params, inputs),
                                backend='gpu')
    _ = flax_apply_jitted(joint_net_vars, example)

    times = []
    for _ in range(num_reps):
        st = time.time()
        _ = flax_apply_jitted(joint_net_vars, example).block_until_ready()
        run_time = time.time() - st
        times.append(run_time)

    median_runtime = np.median(np.array(times))
    return median_runtime


def proposal_analytics(input_features_size,
                       features_array,
                       constraints: Constraints,
                       latency_fn):
    int_features = discretize_features(features_array)

    predicted_lat = latency_fn(int_features).item()

    measured_lat = benchmark_features_array(input_features_size, int_features)

    num_weights = total_weights(input_features_size, int_features)

    print(f"Discretized features: {int_features.tolist()}")
    print(f"predicted {predicted_lat:.6f} measured {measured_lat:.6f} sec, "
          f" {constraints.latency_sec}")
    print(f"num_weights {num_weights}, {constraints.parameters}")

    return dict(predicted_lat=predicted_lat,
                measured_lat=measured_lat,
                num_weights=num_weights)


def create_constraints() -> Constraints:
    min_layers = 5
    max_layers = 20
    max_latency_sec = 0.002
    min_latency_sec = 0.75 * max_latency_sec
    max_parameters = 1_000_000
    min_parameters = 0.5 * max_parameters

    constraints = Constraints(
        MinMax(min_layers, max_layers),
        MinMax(min_latency_sec, max_latency_sec),
        MinMax(min_parameters, max_parameters))
    return constraints


def predict_latencies(predict_fn,
                      params,
                      input_features_size,
                      features_arr):
    feature_tuples = []
    for i_layer in range(len(features_arr)):
        feat_in = input_features_size if i_layer == 0 \
            else features_arr[i_layer - 1]
        feat_out = features_arr[i_layer]

        features_tuple = (feat_in, feat_out)
        feature_tuples.append(features_tuple)

    features_jnp = jnp.array(feature_tuples)
    raw_latencies = predict_fn.apply({'params': params}, features_jnp)
    return raw_latencies


def total_latency_fn(predict_fn, params, input_features_size, features_arr):
    raw_latencies = predict_latencies(predict_fn,
                                      params,
                                      input_features_size,
                                      features_arr)
    latencies = jnp.maximum(raw_latencies, 0)
    total_latency = jnp.sum(latencies)
    return total_latency


def latency_loss_fn(predict_fn,
                    params,
                    input_features_size,
                    features_arr,
                    constraints: Constraints):

    total_latency = total_latency_fn(predict_fn,
                                     params,
                                     input_features_size,
                                     features_arr)
    latency_loss = double_boundary_loss(total_latency,
                                        constraints.latency_sec.min,
                                        constraints.latency_sec.min)
    aux = OrderedDict(total_latency=total_latency, latency_loss=latency_loss)
    return latency_loss, aux


def rem_loss_fn(input_features_size, features_arr, constraints: Constraints):
    weight_nums = []
    for i_layer in range(len(features_arr)):
        feat_in = input_features_size if i_layer == 0 \
            else features_arr[i_layer - 1]
        feat_out = features_arr[i_layer]

        num_weights = calc_weights_leaky(feat_in, feat_out)
        weight_nums.append(num_weights)

    total_num_weights = jnp.sum(jnp.array(weight_nums))
    num_weights_loss = double_boundary_loss(total_num_weights,
                                            constraints.parameters.min,
                                            constraints.parameters.max)

    compactness_array = jnp.maximum(features_arr[1:] - features_arr[:-1], 0)
    compactness_loss = jnp.sum(compactness_array) / \
        jnp.mean(jnp.square(features_arr))

    total_loss = 0.1 * num_weights_loss + 100.0 * compactness_loss

    aux = OrderedDict(
        total_num_weights=total_num_weights, num_weights_loss=num_weights_loss,
        compactness_loss=compactness_loss)

    return total_loss, aux


latency_loss_jit_fn = nn.jit(latency_loss_fn,
                             static_argnums=(2, 4),
                             backend='gpu')
lat_grad_fn = jax.value_and_grad(latency_loss_fn,
                                 argnums=(3,),
                                 has_aux=True)
lat_grad_jit_fn = nn.jit(lat_grad_fn, static_argnums=(2, 4), backend='gpu')

rem_loss_jit_fn = jax.jit(rem_loss_fn, static_argnums=(0, 2))
rem_grad_fn = jax.value_and_grad(rem_loss_fn, argnums=(1,), has_aux=True)
rem_grad_jit_fn = jax.jit(rem_grad_fn, static_argnums=(0, 2), backend='gpu')


def cp_value_and_jacobian(feature_vector: np.ndarray,
                          predict_flax,
                          params,
                          input_features_size,
                          constraints: Constraints) \
        -> Tuple[float, List[float]]:

    (lv, *_), (lg,) = lat_grad_jit_fn(predict_flax,
                                      params,
                                      input_features_size,
                                      feature_vector,
                                      constraints)
    (rv, *_), (rg,) = rem_grad_jit_fn(input_features_size,
                                      feature_vector,
                                      constraints)
    total_val = lv + rv
    total_val_np = np.array(total_val)
    total_grad = lg + rg
    total_grad_np = np.array(total_grad)
    return total_val_np.item(), total_grad_np.tolist()


def print_progress(current_vector):
    print(f"Features {[int(v) for v in current_vector.tolist()]}")
    return False


def optimize(seed_points_np,
             fast,
             cp_value_and_jacobian_args,
             bounds,
             i_outer):

    print(f"Run iteration {i_outer}")

    random_features_np = seed_points_np[i_outer]

    maxiter = 8 if fast else 30

    res = minimize(
        cp_value_and_jacobian,
        random_features_np,
        args=cp_value_and_jacobian_args,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options=dict(maxiter=maxiter),
        callback=None,
        )
    return res


def generate_known_solutions(input_features_size,
                             constraints,
                             total_latency_eval_fn):
    test_lat_dicts = []
    for i_pos in range(4, 8):
        test_features_array = jnp.zeros((constraints.layers.max,),
                                        dtype=jnp.float32) + 512
        test_features_array = test_features_array.at[i_pos].set(-1000)

        print("For", np.array(test_features_array))
        test_lat_dicts.append(proposal_analytics(input_features_size,
                                                 test_features_array,
                                                 constraints,
                                                 total_latency_eval_fn))
    return test_lat_dicts


def gradient_automl_linear(evaluator: Dict[str, Any]):
    constraints = create_constraints()

    # predict_fn = evaluator['predict_fn']
    # module = evaluator['module']
    params = evaluator['params']
    # predict_flax = evaluator['predict_flax'] # TEMPORARY
    from latency_model import LatencyNet
    predict_flax = LatencyNet()  # DEBUG

    input_features_size = 10

    total_latency_eval_fn = partial(total_latency_fn,
                                    predict_flax,
                                    params,
                                    input_features_size)

    test_lat_dicts = []
    if True:
        test_lat_dicts = generate_known_solutions(input_features_size,
                                                  constraints,
                                                  total_latency_eval_fn)

    # gpu = jax.devices("gpu")[0]

    print("Optimizing parameters")

    start_opt_time = time.time()

    CH_MAX = 512

    bounds = [(-CH_MAX, CH_MAX) for _ in range(int(constraints.layers.max))]

    fast = False

    num_outer_iters = 4 if fast else 20

    latin_hypercube = LatinHypercube(int(constraints.layers.max))
    seed_points_np = latin_hypercube.random(num_outer_iters) * (CH_MAX - 1) + 1

    cp_value_and_jacobian_args = (predict_flax, params,
                                  input_features_size,
                                  constraints)
    optimize_partial = partial(optimize,
                               seed_points_np,
                               fast,
                               cp_value_and_jacobian_args,
                               bounds)

    multiprocess = True
    if multiprocess:
        pool_size = 10
        with get_context('spawn').Pool(pool_size) as pool:
            results = pool.map(optimize_partial, range(num_outer_iters))
    else:
        results = []
        for i_outer in range(num_outer_iters):
            res = optimize_partial(i_outer)
            results.append(res)

    print("Optimization done")

    end_opt_time = time.time()
    elapsed_opt_time = end_opt_time - start_opt_time
    print(f"elapsed_opt_time={elapsed_opt_time/60:.1f} min")

    lat_dicts = []
    seed_lat_dicts = []
    for i_outer in range(num_outer_iters):
        random_features_np = seed_points_np[i_outer]
        res = results[i_outer]

        print(">>> random:")
        seed_lat_dict = proposal_analytics(input_features_size,
                                           random_features_np,
                                           constraints,
                                           total_latency_eval_fn)
        print("=== optimized:")
        lat_dict = proposal_analytics(input_features_size,
                                      res.x,
                                      constraints,
                                      total_latency_eval_fn)
        print("<<<")
        lat_dicts.append(lat_dict)
        seed_lat_dicts.append(seed_lat_dict)

        visualize_results(lat_dicts,
                          seed_lat_dicts,
                          test_lat_dicts,
                          constraints)

    print(results)
    for res in results:
        print("---------------------------------------")
        print_progress(res.x)
    print("---------------------------------------")

    # visualize_results(lat_dicts, seed_lat_dicts, test_lat_dicts, constraints)

    print("Automl done")
