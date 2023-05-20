import os
import pickle
import time
import math
from typing import Dict, Any, Iterable, Tuple, List, Union
import numpy as np
from collections import OrderedDict
from functools import partial
from dataclasses import dataclass
from multiprocessing import get_context

import jax
import jax.numpy as jnp
import flax.linen as nn

from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from losses import double_boundary_loss
from constraints import Constraints, MinMax
from visualization import visualize_results
from dataclass_jax import register_pytree_node_dataclass


KERNEL_SIZE_RS = (3, 3)


@register_pytree_node_dataclass
@dataclass(frozen=True)
class Variables:
    features: Union[jax.Array, np.ndarray]
    strides: Union[jax.Array, np.ndarray]

    def flat_numpy(self):
        if isinstance(self.features, jax.Array):
            features = np.array(self.features)
        else:
            features = self.features
        if isinstance(self.strides, jax.Array):
            strides = np.array(self.strides)
        else:
            strides = self.strides
        concat = np.concatenate((features, strides), axis=0)
        assert len(concat.shape) == 1
        return concat

    @classmethod
    def from_flat(cls, flat_vars: np.ndarray) -> 'Variables':
        features = flat_vars[:len(flat_vars)//2]
        strides = flat_vars[len(flat_vars)//2:]
        return cls(features, strides)

    def __add__(self, other: 'Variables'):
        return self.__class__(self.features + other.features,
                              self.strides + other.strides)


def calc_weights_leaky(fin, fout, kernel_h, kernel_w):
    negative_slope = 1e-2
    result = nn.leaky_relu(fin + 1, negative_slope) * \
        nn.leaky_relu(fout, negative_slope) * \
        nn.leaky_relu(kernel_h, negative_slope) * \
        nn.leaky_relu(kernel_w, negative_slope)
    return result


def calc_weights_precise(fin, fout, kernel_h, kernel_w):
    result = nn.relu(fin + 1) * \
        nn.relu(fout) * \
        nn.relu(kernel_h) * \
        nn.relu(kernel_w)
    return result


def total_weights(input_shape_nhwc: Tuple[int, ...], variables: Variables) \
        -> int:

    weight_nums = []
    for i_layer in range(len(variables.features)):
        feat_in = input_shape_nhwc[3] if i_layer == 0 \
            else variables.features[i_layer - 1]
        feat_out = variables.features[i_layer]

        num_weights = calc_weights_precise(feat_in, feat_out, *KERNEL_SIZE_RS)
        weight_nums.append(num_weights)
    return np.sum(np.array(weight_nums)).item()


class DCN(nn.Module):
    hidden_feat: Union[Iterable[int], np.ndarray]
    hidden_strides: Union[Iterable[int], np.ndarray]

    @nn.compact
    def __call__(self, x):
        for out_f, strides in zip(self.hidden_feat, self.hidden_strides):
            conv = nn.Conv(out_f.item(),
                           kernel_size=KERNEL_SIZE_RS,
                           strides=strides.item())
            x = conv(x)
            x = nn.relu(x)
        return x


def discretize_features(variables: Variables):
    ch_step = 4
    min_stride = 1
    max_stride = 2
    features_np = np.array(variables.features)
    strides_np = np.array(variables.strides)
    res_features = []
    res_strides = []
    for feature, stride in zip(features_np, strides_np):
        feature_int = int(feature)
        stride_int = int(round(stride))
        stride_int = min_stride if stride_int < min_stride \
            else max_stride if stride_int > max_stride else stride_int
        if feature_int <= 0:
            break
        feature_disc = int(math.ceil(feature_int / ch_step) * ch_step)
        if feature_disc <= 0:
            print("ERROR: feature_disc <= 0")
        res_features.append(feature_disc)
        res_strides.append(stride_int)
    return Variables(
        np.array(res_features, dtype=np.int32),
        np.array(res_strides, dtype=np.int32))


def benchmark_variables(input_shape_nhwc: Tuple[int, ...],
                        variables: Variables,
                        num_reps=5):

    disc_variables = discretize_features(variables)

    joint_net = DCN(disc_variables.features, disc_variables.strides)
    example = jnp.ones(input_shape_nhwc)
    joint_net_vars = joint_net.init(jax.random.PRNGKey(44), example)

    flax_apply_jitted = jax.jit(lambda params, inputs:
                                joint_net.apply(params, inputs),
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


def proposal_analytics(input_shape_nhwc: Tuple[int, ...],
                       variables: Variables,
                       constraints: Constraints,
                       latency_fn):
    disc_variables = discretize_features(variables)

    predicted_lat = latency_fn(disc_variables).item()

    measured_lat = benchmark_variables(input_shape_nhwc, disc_variables)

    num_weights = total_weights(input_shape_nhwc, disc_variables)

    print(f"Discretized features: {disc_variables.features.tolist()} "
          f"{disc_variables.strides.tolist()}")
    print(f"predicted {predicted_lat:.6f} measured {measured_lat:.6f} sec, "
          f" {constraints.latency_sec}")
    print(f"num_weights {num_weights}, {constraints.parameters}")

    return dict(predicted_lat=predicted_lat,
                measured_lat=measured_lat,
                num_weights=num_weights)


def create_constraints() -> Constraints:
    min_layers = 5
    max_layers = 20
    max_latency_sec = 0.01
    min_latency_sec = 0.75 * max_latency_sec
    max_parameters = 10_000_000
    min_parameters = 0.5 * max_parameters

    constraints = Constraints(
        MinMax(min_layers, max_layers),
        MinMax(min_latency_sec, max_latency_sec),
        MinMax(min_parameters, max_parameters))
    return constraints


def predict_latencies(predict_fn,
                      params,
                      input_shape_nhwc: Tuple[int, ...],
                      variables: Variables):
    batch = input_shape_nhwc[0]
    reso_hw = input_shape_nhwc[1:3]
    ch_in = input_shape_nhwc[3]

    feature_tuples = []
    for i_layer in range(len(variables.features)):
        feat_in = ch_in if i_layer == 0 \
            else variables.features[i_layer - 1]
        feat_out = variables.features[i_layer]
        stride = variables.strides[i_layer]

        features_tuple = (feat_in, reso_hw[0], batch, reso_hw[1],
                          feat_out, *KERNEL_SIZE_RS, stride, stride)
        feature_tuples.append(features_tuple)

        # if stride.item() > 1.5:
        #     reso_hw = reso_hw / 2

        reso_hw = jax.lax.cond(stride > 1.5,
                               lambda x: tuple([v//2 for v in x]),
                               lambda x: x, reso_hw)

    features_jnp = jnp.array(feature_tuples)
    raw_latencies = predict_fn.apply({'params': params}, features_jnp)
    return raw_latencies


def total_latency_fn(predict_fn,
                     params,
                     input_shape_nhwc: Tuple[int, ...],
                     variables: Variables):
    raw_latencies = predict_latencies(predict_fn,
                                      params,
                                      input_shape_nhwc,
                                      variables)
    latencies = jnp.maximum(raw_latencies, 0)
    total_latency = jnp.sum(latencies)
    return total_latency


def latency_loss_fn(predict_fn,
                    params,
                    input_shape_nhwc: Tuple[int, ...],
                    variables: Variables,
                    constraints: Constraints):

    raw_latencies = predict_latencies(predict_fn,
                                      params,
                                      input_shape_nhwc,
                                      variables)
    latencies = jnp.maximum(raw_latencies, 0)
    total_latency = jnp.sum(latencies)
    latency_loss = double_boundary_loss(total_latency,
                                        constraints.latency_sec.min,
                                        constraints.latency_sec.max)
    aux = OrderedDict(total_latency=total_latency,
                      latency_loss=latency_loss,
                      raw_latencies=raw_latencies)
    return latency_loss, aux


def rem_loss_fn(input_shape_nhwc: Tuple[int, ...],
                variables: Variables,
                constraints: Constraints):
    ch_in = input_shape_nhwc[3]

    weight_nums = []
    for i_layer in range(int(constraints.layers.max)):
        feat_in = ch_in if i_layer == 0 \
                else variables.features[i_layer - 1]
        feat_out = variables.features[i_layer]

        num_weights = calc_weights_leaky(feat_in, feat_out, *KERNEL_SIZE_RS)
        weight_nums.append(num_weights)

    total_num_weights = jnp.sum(jnp.array(weight_nums))
    num_weights_loss = double_boundary_loss(total_num_weights,
                                            constraints.parameters.min,
                                            constraints.parameters.max)

    # compactness_array = jnp.maximum(features_arr[1:] - features_arr[:-1], 0)
    # compactness_loss = jnp.sum(compactness_array) / jnp.mean(jnp.square(features_arr))
    compactness_loss = 0.0

    total_loss = 0.1 * num_weights_loss + 100.0 * compactness_loss
    # total_loss = 100.0 * compactness_loss

    aux = OrderedDict(
        total_num_weights=total_num_weights,
        num_weights_loss=num_weights_loss,
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


def cp_value_and_jacobian(flat_variables: np.ndarray,
                          predict_flax,
                          params,
                          input_shape_nhwc: Tuple[int, ...],
                          constraints: Constraints) \
        -> Tuple[float, List[float]]:

    variables = Variables.from_flat(flat_variables)

    (lv, *_), (lg,) = lat_grad_jit_fn(predict_flax,
                                      params,
                                      input_shape_nhwc,
                                      variables,
                                      constraints)
    (rv, *_), (rg,) = rem_grad_jit_fn(input_shape_nhwc,
                                      variables,
                                      constraints)
    total_val = lv + rv
    total_val_np = np.array(total_val)
    total_grad: Variables = lg + rg
    total_grad_flat_np = total_grad.flat_numpy()
    return total_val_np.item(), total_grad_flat_np.tolist()


def optimize(seed_points: np.ndarray,
             fast,
             cp_value_and_jacobian_args,
             bounds,
             i_outer):

    print(f"Run iteration {i_outer}")

    flat_seed_vars_np = seed_points[i_outer]

    maxiter = 8 if fast else 30

    res = minimize(
        cp_value_and_jacobian,
        flat_seed_vars_np,
        args=cp_value_and_jacobian_args,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options=dict(maxiter=maxiter),
        callback=None,
        )
    return res


def print_results(current_vector: np.ndarray):
    print(f"Features & strides {[int(v) for v in current_vector.tolist()]}")


def gradient_automl_conv2d(evaluator):
    constraints = create_constraints()

    params = evaluator['params']
    # predict_flax = evaluator['predict_flax'] # TEMPORARY
    from latency_model import LatencyNet
    predict_flax = LatencyNet()  # DEBUG

    input_shape_nhwc = (1, 160, 160, 4)
    init_features = 300
    init_strides = 1
    test_features_array = jnp.zeros((constraints.layers.max,),
                                    dtype=jnp.float32) + init_features
    # stride can be 1 or 2
    test_strides_array = jnp.zeros((constraints.layers.max,),
                                   dtype=jnp.float32) + init_strides
    test_variables = Variables(test_features_array, test_strides_array)

    total_latency_eval_fn = partial(total_latency_fn,
                                    predict_flax,
                                    params,
                                    input_shape_nhwc)

    seed_lat_dict = proposal_analytics(input_shape_nhwc,
                                       test_variables,
                                       constraints,
                                       total_latency_eval_fn)

    #     "Tensor3DSpecs_c",
    #     "Tensor3DSpecs_h",
    #     "Tensor3DSpecs_n",
    #     "Tensor3DSpecs_w",
    #     "ConvSpecs_k",
    #     "ConvSpecs_r",
    #     "ConvSpecs_s",
    #     "ConvSpecs_u",
    #     "ConvSpecs_v"

    print("Optimizing parameters")

    start_opt_time = time.time()

    CH_MAX = 256

    bounds_features = [(-CH_MAX, CH_MAX)
                       for _ in range(int(constraints.layers.max))]
    bounds_strides = [(1, 2) for _ in range(int(constraints.layers.max))]
    bounds = bounds_features + bounds_strides

    fast = True

    num_outer_iters = 5 if fast else 5

    latin_hypercube_feat = LatinHypercube(int(constraints.layers.max))
    seed_features_np = latin_hypercube_feat.random(num_outer_iters) * \
        (CH_MAX - 1) + 1
    latin_hypercube_strd = LatinHypercube(int(constraints.layers.max))
    seed_strides_np = latin_hypercube_strd.random(num_outer_iters) + 1
    seed_flat_variables_np = np.concatenate((seed_features_np,
                                            seed_strides_np), axis=-1)

    cp_value_and_jacobian_args = (predict_flax, params,
                                  input_shape_nhwc,
                                  constraints)
    optimize_partial = partial(optimize,
                               seed_flat_variables_np,
                               fast,
                               cp_value_and_jacobian_args,
                               bounds)

    results_cache_path = "results_cache_conv2d.pickle"
    if os.path.exists(results_cache_path):
        with open(results_cache_path, "rb") as f:
            results = pickle.load(f)
    else:
        multiprocess = True
        if multiprocess:
            pool_size = 10
            with get_context('spawn').Pool(pool_size) as pool:
                results = pool.map(optimize_partial, range(num_outer_iters))
        else:
            results = []
            for i_outer in range(num_outer_iters):
                res = optimize_partial(i_outer)
                print("-"*30)
                print(res)
                results.append(res)
        with open(results_cache_path, "wb") as f:
            pickle.dump(results, f)

    print("Optimization done")

    end_opt_time = time.time()
    elapsed_opt_time = end_opt_time - start_opt_time
    print(f"elapsed_opt_time={elapsed_opt_time/60:.1f} min")

    lat_dicts = []
    seed_lat_dicts = []
    for i_outer in range(num_outer_iters):
        flat_seed_vars_np = seed_flat_variables_np[i_outer]
        seed_variables = Variables.from_flat(flat_seed_vars_np)
        res = results[i_outer]

        print(">>> seed:")
        seed_lat_dict = proposal_analytics(input_shape_nhwc,
                                           seed_variables,
                                           constraints,
                                           total_latency_eval_fn)
        print("=== optimized:")
        optimized_variables = Variables.from_flat(res.x)
        lat_dict = proposal_analytics(input_shape_nhwc,
                                      optimized_variables,
                                      constraints,
                                      total_latency_eval_fn)
        print("<<<")
        lat_dicts.append(lat_dict)
        seed_lat_dicts.append(seed_lat_dict)

        visualize_results(lat_dicts,
                          seed_lat_dicts,
                          [seed_lat_dict],
                          constraints,
                          'conv2d')

    print(results)

    for res in results:
        print("---------------------------------------")
        print_results(res.x)
    print("---------------------------------------")

    print("Automl done")
