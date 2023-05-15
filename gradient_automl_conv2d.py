
import time
from typing import Dict, Any, Iterable, Tuple
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import jax
import jax.numpy as jnp
import flax.linen as nn

from losses import double_boundary_loss



def calc_weights_leaky(fin, fout, kernel_h, kernel_w):
    negative_slope = 1e-2
    result = nn.leaky_relu(fin + 1, negative_slope) * nn.leaky_relu(fout, negative_slope) * \
        nn.leaky_relu(kernel_h, negative_slope) * nn.leaky_relu(kernel_w, negative_slope)
    return result


class DCN(nn.Module):
    hidden_feat: Iterable[int]
    hidden_strides: Iterable[int]

    @nn.compact
    def __call__(self, x):
        for out_f, strides in zip(self.hidden_feat, self.hidden_strides):
            conv = nn.Conv(out_f, kernel_size=(3, 3), strides=strides.item())
            x = conv(x)
            x = nn.relu(x)
        return x


def benchmark_solution(input_shape_nhwc: Tuple[int], features_array, strides_array):
    step = 4
    int_features = (np.ceil(np.array(features_array) / step) * step).astype(np.int32)
    int_features[int_features < step] = step
    
    int_strides = np.round(np.array(strides_array))
    int_strides = np.clip(int_strides, 1, 2).astype(np.int32)

    assert int_features.shape == int_strides.shape
    
    joint_net = DCN(int_features, int_strides)
    example = jnp.ones(input_shape_nhwc)
    joint_net_vars = joint_net.init(jax.random.PRNGKey(44), example)

    flax_apply_jitted = jax.jit(lambda params, inputs: joint_net.apply(params, inputs))
    pred = flax_apply_jitted(joint_net_vars, example)

    for _ in range(10):
        st = time.time()
        pred = flax_apply_jitted(joint_net_vars, example).block_until_ready()
        print(time.time() - st)


def gradient_automl_conv2d(evaluator):
    min_layers = 5
    max_layers = 10
    max_latency_sec = 0.005
    min_latency_sec = 0.75 * max_latency_sec
    max_parameters = 10_000_000
    min_parameters = 0.5 * max_parameters
    
    input_shape_nhwc = (1, 160, 160, 4)
    init_features = 300
    init_strides = 1
    features_array = jnp.zeros((max_layers,), dtype=jnp.float32) + init_features
    strides_array = jnp.zeros((max_layers,), dtype=jnp.float32) + init_strides # stride can be 1 or 2

    benchmark_solution(input_shape_nhwc, features_array, strides_array)

    #     "Tensor3DSpecs_c",
    #     "Tensor3DSpecs_h",
    #     "Tensor3DSpecs_n",
    #     "Tensor3DSpecs_w",
    #     "ConvSpecs_k",
    #     "ConvSpecs_r",
    #     "ConvSpecs_s",
    #     "ConvSpecs_u",
    #     "ConvSpecs_v"

    def predict_latencies(predict_fn, params, features_arr, strides_arr):
        feature_tuples = []
        reso_hw = input_shape_nhwc[1:3]
        batch = input_shape_nhwc[0]
        ch_in = input_shape_nhwc[3]

        for i_layer in range(max_layers):
            feat_in = ch_in if i_layer == 0 else features_arr[i_layer - 1]
            feat_out = features_arr[i_layer]
            stride = strides_arr[i_layer]

            features_tuple = (feat_in, reso_hw[0], batch, reso_hw[1],
                              feat_out, 3, 3, stride, stride)
            feature_tuples.append(features_tuple)

            # if stride.item() > 1.5:
            #     reso_hw = reso_hw / 2
            
            reso_hw = jax.lax.cond(stride > 1.5, lambda x: tuple([v//2 for v in x]), lambda x: x, reso_hw)

        features_jnp = jnp.array(feature_tuples)
        raw_latencies = predict_fn.apply({'params': params}, features_jnp)
        return raw_latencies
    
    def rem_loss_fn(features_arr, strides_arr):
        ch_in = input_shape_nhwc[3]

        weight_nums = []
        for i_layer in range(max_layers):
            feat_in = ch_in if i_layer == 0 else features_arr[i_layer - 1]
            feat_out = features_arr[i_layer]

            num_weights = calc_weights_leaky(feat_in, feat_out, 3, 3)
            weight_nums.append(num_weights)

        total_num_weights = jnp.sum(jnp.array(weight_nums))
        num_weights_loss = double_boundary_loss(total_num_weights, min_parameters, max_parameters)

        # compactness_array = jnp.maximum(features_arr[1:] - features_arr[:-1], 0)
        # compactness_loss = jnp.sum(compactness_array) / jnp.mean(jnp.square(features_arr))
        compactness_loss = 0.0

        total_loss = 0.1 * num_weights_loss + 100.0 * compactness_loss
        # total_loss = 100.0 * compactness_loss

        aux = OrderedDict(
            total_num_weights=total_num_weights, num_weights_loss=num_weights_loss,
            compactness_loss=compactness_loss)

        return total_loss, aux
    
    def latency_loss_fn(predict_fn, params, features_arr, strides_arr):
        raw_latencies = predict_latencies(predict_fn, params, features_arr, strides_arr)
        latencies = jnp.maximum(raw_latencies, 0)
        total_latency = jnp.sum(latencies)
        latency_loss = double_boundary_loss(total_latency, min_latency_sec, max_latency_sec)
        aux = OrderedDict(total_latency=total_latency, latency_loss=latency_loss,
            raw_latencies=raw_latencies)
        return latency_loss, aux

    params = evaluator['params']
    predict_flax = evaluator['predict_flax']

    lat_loss, lat_aux_dict = latency_loss_fn(predict_flax, params, features_array, strides_array)
    latency_loss_jit_fn = nn.jit(latency_loss_fn)
    lat_loss, lat_aux_dict = latency_loss_jit_fn(predict_flax, params, features_array, strides_array)
    lat_grad_fn = nn.jit(jax.value_and_grad(latency_loss_jit_fn, argnums=(2, 3), has_aux=True))
    llg, lat_aux_dict = lat_grad_fn(predict_flax, params, features_array, strides_array)

    raw_latencies = predict_latencies(predict_flax, params, features_array, strides_array)

    tlv = rem_loss_fn(features_array, strides_array)
    
    rem_loss_jit_fn = jax.jit(rem_loss_fn)
    tljv = rem_loss_jit_fn(features_array, strides_array)

    rem_grad_fn = jax.value_and_grad(rem_loss_jit_fn, argnums=(0,), has_aux=True)
    gfnv = rem_grad_fn(features_array, strides_array)

    rem_grad_fn_jit = jax.jit(rem_grad_fn)
    gfnjv = rem_grad_fn_jit(features_array, strides_array)

    print("Optimizing parameters")

    for i_step in tqdm(range(100)):
        st = time.time()
        (lat_loss, lat_aux_dict), (feat_grad, strd_grad) = lat_grad_fn(predict_flax, params, features_array, strides_array)
        lat_loss.block_until_ready()
        print(time.time() - st)
        st = time.time()
        (rem_loss, rem_aux_dict), (rem_feat_grad,) = rem_grad_fn_jit(features_array, strides_array)
        rem_loss.block_until_ready()
        print(time.time() - st)
        total_feat_grad = feat_grad + rem_feat_grad
        feat_grad_scaled_mimimize = - 1e+4 * np.array(total_feat_grad)
        strd_grad_scaled_mimimize = - 5e+1 * np.array(strd_grad)
        print(f"--- step={i_step} lat_loss={lat_loss.item():.6f} rem_loss={rem_loss.item():.6f}")
        lat_aux_dict = {k: v.item() if v.shape == () else v for k, v in lat_aux_dict.items()}
        rem_aux_dict = {k: v.item() if v.shape == () else v for k, v in rem_aux_dict.items()}
        print(f"lat_aux_dict={lat_aux_dict}")
        print(f"rem_aux_dict={rem_aux_dict}")
        print(f"feat_grad={feat_grad_scaled_mimimize}")
        print(f"strd_grad={strd_grad_scaled_mimimize}")
        if jnp.mean(jnp.abs(feat_grad_scaled_mimimize)).item() < 1e-6:
            # print("Grads are zero. Early stopping.")
            # break
            pass
        features_array += feat_grad_scaled_mimimize
        strides_array += strd_grad_scaled_mimimize
        print(f"features_array={features_array.astype(np.int32)}")
        print(f"strides_array={strides_array}")
        benchmark_solution(input_shape_nhwc, features_array, strides_array)

    print(features_array, strides_array)

    benchmark_solution(input_shape_nhwc, features_array, strides_array)

    print("Automl done")   