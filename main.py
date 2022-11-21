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
        k = np_rng.integers(16, 32)
        c = np_rng.integers(16, 32)
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

    # def __init__(self, specs: ConvSpecs):
    #     super().__init__()
    #     self.specs = specs

    @nn.compact
    def __call__(self, x):
        print(self.specs)
        print(x.shape)
        conv = nn.Conv(features=self.specs.r,
            kernel_size=(self.specs.r, self.specs.s),
            strides=(self.specs.u, self.specs.v))
        x = conv(x)
        x = nn.relu(x)
        return x


class Operator:
    def __init__(self, inp_tensor: jnp.DeviceArray, op: nn.Module, rng_key):
        self.inp_tensor = inp_tensor
        self.op = op
        self.rng_key = rng_key

    def benchmark(self):
        # cnn = self.op()
        shape = self.inp_tensor.shape
        variables = self.op.init(self.rng_key, jnp.ones(shape))
        start_time = time.time()
        pred = self.op.apply(variables, self.inp_tensor)
        pred.wait_until_done()
        duration_s = time.time() - start_time
        return duration_s


def OperatorFactory(device, seed=42) -> Iterable[Operator]:
    rng_key = jax.random.PRNGKey(seed=seed)
    np_rng = np.random.default_rng(np.asarray(rng_key))
    while True:
        # conv_op_specs = ConvOpSpecs.get_random(np_rng)
        conv_op_specs = ConvOpSpecs(TensorSpecs(28, 28, 3), ConvSpecs(16, 3, 3, 1, 1))
        shape = conv_op_specs.tensor_specs
        inp_cpu = jax.random.uniform(rng_key, shape=(1, shape.h, shape.w, shape.c))
        print(type(inp_cpu))
        print(inp_cpu.device_buffer.device())
        inp = jax.device_put(inp_cpu, device)
        print(type(inp))
        print(inp.device_buffer.device())
        layer = OneConvLayer(conv_op_specs.conv_specs)
        op = Operator(inp, layer, rng_key)
        yield op


def main():
    cpus = jax.devices("cpu")
    gpus = jax.devices("gpu")
    gpu = gpus[0]

    factory = OperatorFactory(gpu)
    op = next(factory)
    print(op)
    print(op.benchmark())

    print("Done")


if __name__ == "__main__":
    main()
