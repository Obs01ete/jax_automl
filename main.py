import time
from typing import Iterable
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass


@dataclass(frozen=True)
class ConvSpecs:
    k: int # ch_out
    r: int # kernel_h
    s: int # kernel_w
    u: int # stride_h
    v: int # stride_w


@dataclass(frozen=True)
class TensorSpecs:
    h: int # tensor height
    w: int # tensor width
    c: int # ch_in


class ConvOpSpecs:
    def __init__(self, tensor_specs: TensorSpecs, conv_specs: ConvSpecs):
        self.tensor_specs = tensor_specs
        self.conv_specs = conv_specs

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
        ts = TensorSpecs(h, w, c)
        cs =  ConvSpecs(k, r, s, u, v)
        os = cls(ts, cs)
        return os



class OneConvLayer(nn.Module):
    specs: ConvSpecs

    def setup(self):
        self.linear = nn.Dense(features=self.specs.r)

    # @nn.compact
    def __call__(self, x):
        # print(self.specs)
        # print(x.shape)
        # conv = nn.Conv(features=self.specs.r,
        #     kernel_size=(self.specs.r, self.specs.s),
        #     strides=(self.specs.u, self.specs.v))
        # x = conv(x)
        x = self.linear(x)
        x = nn.relu(x)
        return x


class Operator:
    def __init__(self, tensor_shape: TensorSpecs, op: OneConvLayer, rng_key, device: str):
        self.tensor_shape = tensor_shape
        self.op = op
        self.rng_key = rng_key
        self.device = device

    def benchmark(self):
        print(self.tensor_shape, self.op.specs)
        shape = self.tensor_shape
        tensor_shape = (1, shape.h, shape.w, shape.c)
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
        print("median_s", median_s)
        compile_time_s = durations_s[0] - median_s
        print("compile_time_s", compile_time_s)
        return median_s


def OperatorFactory(device, seed=42) -> Iterable[Operator]:
    rng_key = jax.random.PRNGKey(seed=seed)
    np_rng = np.random.default_rng(np.asarray(rng_key))
    while True:
        conv_op_specs = ConvOpSpecs.get_random(np_rng)
        # conv_op_specs = ConvOpSpecs(TensorSpecs(28, 28, 3), ConvSpecs(16, 3, 3, 1, 1))
        layer = OneConvLayer(conv_op_specs.conv_specs)
        op = Operator(conv_op_specs.tensor_specs, layer, rng_key, device)
        rng_key, _ = jax.random.split(rng_key)
        yield op


def main():
    cpus = jax.devices("cpu")
    gpus = jax.devices("gpu")
    gpu = gpus[0]

    factory = OperatorFactory(gpu)
    for _ in range(10):
        op = next(factory)
        # print(op)
        op.benchmark()

    print("Done")


if __name__ == "__main__":
    main()
