import os
import time
import json
from typing import Iterable, Union, Callable, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import dataclasses
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm


class Linearizable(ABC):
    @abstractmethod
    def linearize(self) -> Tuple[int]:
        pass


@dataclass(frozen=True)
class ConvSpecs:
    k: int # ch_out
    r: int # kernel_h
    s: int # kernel_w
    u: int # stride_h
    v: int # stride_w


@dataclass(frozen=True)
class LinearSpecs:
    k: int # ch_out


@dataclass(frozen=True)
class Tensor3DSpecs(Linearizable):
    h: int # tensor height
    w: int # tensor width
    c: int # ch_in

    def linearize(self):
        return (self.h, self.w, self.c)


@dataclass(frozen=True)
class Tensor1DSpecs(Linearizable):
    f: int # features (neurons)

    def linearize(self):
        return (self.f,)


class ConvOpSpecs:
    def __init__(self, tensor_specs: Tensor3DSpecs, op_specs: ConvSpecs):
        self.tensor_specs = tensor_specs
        self.op_specs = op_specs

    @classmethod
    def get_random(cls, np_rng: np.random._generator.Generator):
        h, w = np_rng.choice(list(range(4, 256, 4)), size=(2,))
        k, c = np_rng.choice(list(range(4, 512, 4)), size=(2,))
        # h, w = 512, 512
        # k, c = 128, np_rng.choice(list(range(4, 512, 4)))
        r = np_rng.choice((1, 3, 5))
        s = np_rng.choice((1, 3, 5))
        u = np_rng.choice((1, 2))
        v = np_rng.choice((1, 2))
        ts = Tensor3DSpecs(h, w, c)
        cs =  ConvSpecs(k, r, s, u, v)
        os = cls(ts, cs)
        return os


class LinearOpSpecs:
    def __init__(self, tensor_specs: Tensor1DSpecs, op_specs: LinearSpecs):
        self.tensor_specs = tensor_specs
        self.op_specs = op_specs

    @classmethod
    def get_random(cls, np_rng: np.random._generator.Generator):
        # fi, fo = np_rng.choice(list(range(16, 4096, 16)), size=(2,))
        max_features = 4096
        step = 16
        fifo = np_rng.exponential(scale=max_features//4, size=(2,))
        fifo = np.remainder(fifo, max_features).astype(np.int32)
        fifo = (np.ceil(fifo / step) * step).astype(np.int32)
        fi, fo = (int(v) for v in fifo)
        # fi, fo = 512, 1024
        ts = Tensor1DSpecs(fi)
        cs = LinearSpecs(fo)
        os = cls(ts, cs)
        return os


class OneConvLayer(nn.Module):
    specs: ConvSpecs

    def setup(self):
        self.conv = nn.Conv(features=self.specs.r,
            kernel_size=(self.specs.r, self.specs.s),
            strides=(self.specs.u, self.specs.v))

    # @nn.compact
    def __call__(self, x):
        x = self.conv(x)
        x = nn.relu(x)
        return x


class OneLinearLayer(nn.Module):
    specs: LinearSpecs

    def setup(self):
        self.linear = nn.Dense(features=self.specs.k)

    # @nn.compact
    def __call__(self, x):
        x = self.linear(x)
        x = nn.relu(x)
        return x


class Operator:
    def __init__(self, tensor_shape: Linearizable,
                 op: Callable, rng_key, device: str):
        self.tensor_shape = tensor_shape
        self.op = op
        self.rng_key = rng_key
        self.device = device
    
    def get_params(self):
        return self.tensor_shape, self.op.specs

    def benchmark(self):
        print(self.tensor_shape, self.op.specs)
        shape = self.tensor_shape
        batch = 1000
        tensor_shape = (batch, *shape.linearize())
        def _init():
            return self.op.init(self.rng_key, jnp.ones(tensor_shape))
        variables = _init()
        # model = self.op.bind(variables, mutable=False)
        def _create():
            inp_cpu = jax.random.uniform(self.rng_key, shape=tensor_shape)
            inp_tensor = jax.device_put(inp_cpu, self.device)
            return inp_tensor
        forward = jax.jit(self.op.apply)
        def _forward(*args):
            return forward(*args).block_until_ready()
        durations_s = []
        for i in range(5):
            inp_tensor = _create()
            start_time = time.time()
            result = _forward(variables, inp_tensor)
            duration_s = time.time() - start_time
            durations_s.append(duration_s)
            # print(result[0, 0, 0, :])
            self.rng_key, _ = jax.random.split(self.rng_key)
        print("durations_s", durations_s)
        median_s = np.median(durations_s)
        print(f"--- median_s={median_s:.6f}")
        compile_time_s = durations_s[0] - median_s
        print("compile_time_s", compile_time_s)
        return median_s


def OperatorFactory(device, op_type, seed=42) -> Iterable[Operator]:
    rng_key = jax.random.PRNGKey(seed=seed)
    np_rng = np.random.default_rng(np.asarray(rng_key))
    while True:
        op_specs_class = {'linear': LinearOpSpecs, 'conv2d': ConvOpSpecs}[op_type]
        random_op_specs = op_specs_class.get_random(np_rng)
        # conv_op_specs = ConvOpSpecs(TensorSpecs(28, 28, 3), ConvSpecs(16, 3, 3, 1, 1))
        one_layer_class = {'linear': OneLinearLayer, 'conv2d': OneConvLayer}[op_type]
        layer = one_layer_class(random_op_specs.op_specs)
        op = Operator(random_op_specs.tensor_specs, layer, rng_key, device)
        rng_key, _ = jax.random.split(rng_key)
        yield op


def create_dataset(device, op_type, num_samples):
    factory = OperatorFactory(device, op_type)
    measurement_list = []
    common_feature_names = None
    for _ in tqdm(range(num_samples)):
        op = next(factory)
        # print(op)
        params = op.get_params()
        feature_names = []
        features = []
        for group in params:
            group_name = group.__class__.__name__
            group_dict = dataclasses.asdict(group)
            keys = sorted(list(group_dict.keys()))
            for key in keys:
                feature_name = f"{group_name}_{key}"
                value = group_dict[key]
                feature_names.append(feature_name)
                features.append(value)
        if common_feature_names is None:
            common_feature_names = feature_names
        if feature_names != common_feature_names:
            print("Error: different features")
        latency = op.benchmark()
        measurement_list.append(dict(features=features, target=latency))
    return dict(dataset=measurement_list, feature_names=common_feature_names)


def main():
    cpus = jax.devices("cpu")
    gpus = jax.devices("gpu")
    gpu = gpus[0]

    op_type = 'linear' # 'linear' 'conv2d'
    dataset_name = f"{op_type}_data.json"
    if os.path.exists(dataset_name):
        with open(dataset_name, "r") as f:
            dataset = json.load(f)
    else:
        num_samples = 1000
        print(f"Measuring {num_samples} samples")
        time_start = time.time()
        dataset = create_dataset(gpu, op_type, num_samples)
        gen_wall_time = time.time() - time_start
        print(f"Generated in {int(gen_wall_time)} seconds")
        with open(dataset_name, "w") as f:
            json.dump(dataset, f, indent=4)

    # print(dataset)

    print("Done")


if __name__ == "__main__":
    main()
