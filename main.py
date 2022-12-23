import os
import time
import json
from typing import Iterable, Union, Callable, Tuple
import numpy as np
import dataclasses
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)
import optax
from flax.training import train_state


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
        fifo = np_rng.exponential(scale=max_features//2, size=(2,))
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
        num_samples = 1000
        print(f"Measuring {num_samples} samples")
        time_start = time.time()
        dataset = create_dataset(gpu, op_type, num_samples)
        gen_wall_time = time.time() - time_start
        print(f"Generated in {int(gen_wall_time)} seconds")
        with open(dataset_name, "w") as f:
            json.dump(dataset, f, indent=4)

    return dataset


class LatencyNet(nn.Module):

    def setup(self):
        ch = 128
        self.linear1 = nn.Dense(features=ch)
        self.linear2 = nn.Dense(features=ch)
        self.linear3 = nn.Dense(features=ch)
        self.linear4 = nn.Dense(features=ch)
        self.linear5 = nn.Dense(features=ch)
        self.linear6 = nn.Dense(features=ch)
        self.linear7 = nn.Dense(features=ch)
        self.linear8 = nn.Dense(features=ch)
        self.linear9 = nn.Dense(features=1)

    @nn.compact
    def __call__(self, x):
        x = 1e-3 * x
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        x = nn.relu(self.linear3(x))
        x = nn.relu(self.linear4(x))
        x = nn.relu(self.linear5(x))
        x = nn.relu(self.linear6(x))
        x = nn.relu(self.linear7(x))
        x = nn.relu(self.linear8(x))
        x = self.linear9(x)
        x = 1e-3 * x
        x = x.squeeze(1)
        return x


def save_checkpoint(ckpt_path, state, epoch):
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))


def load_checkpoint(ckpt_path, state):
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)


def init_train_state(
    model, random_key, shape, learning_rate
) -> train_state.TrainState:
    variables = model.init(random_key, jnp.ones(shape))
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn = model.apply,
        tx=optimizer,
        params=variables['params']
    )


def total_loss(*, pred, label):
    smoothener_sec = 0.001
    total = mape_metric(pred=pred, label=label, smoothener=smoothener_sec)
    return total


def mape_metric(*, pred, label, smoothener=0.0):
    losses = jnp.abs(pred - label) / (label + smoothener)

    # losses_aval = losses.aval
    # print(losses_aval.shape)
    # print(losses_aval)
    # losses_np = np.asarray(losses_aval)
    # print(losses_np.shape)
    # print(type(losses_np))

    return losses.mean()


def compute_metrics(*, pred, label):
    loss = total_loss(pred=pred, label=label)
    mape = mape_metric(pred=pred, label=label)
    metrics = {
        'loss': loss,
        'mape': mape,
    }
    return metrics


def accumulate_metrics(metrics):
    metrics = jax.device_get(metrics)
    all_metrics = {
        k: np.mean([metric[k] for metric in metrics])
        for k in metrics[0].keys()
    } if len(metrics) > 0 else {}
    return all_metrics


@jax.jit
def train_step(
    state: train_state.TrainState, batch: jnp.ndarray
):
    feature, label = batch

    def loss_fn(params):
        pred = state.apply_fn({'params': params}, feature)
        loss = total_loss(pred=pred, label=label)
        return loss, pred

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, pred), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(pred=pred, label=label)
    return state, metrics, pred


class LatencyModelTrainer:
    def __init__(self, dataset):
        self.features = np.array([r['features'] for r in dataset['dataset']])
        self.targets = np.array([r['target'] for r in dataset['dataset']])

        self.batch_size = 128
        learning_rate = 1e-5

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.features, self.targets))
        # lin_feat = np.tile(np.expand_dims(np.linspace(0.01/1000, 0.01, 1000), 1), (1, 2))
        # lin_targets = np.linspace(0.01/1000, 0.01, 1000)
        # self.train_dataset = tf.data.Dataset.from_tensor_slices((lin_feat, lin_targets))

        self.net = LatencyNet()

        self.rng = jax.random.PRNGKey(43)
        self.state = init_train_state(
           self.net, self.rng, (self.batch_size, self.features.shape[1]), learning_rate)

        print("")
        
    def train(self):
        num_train_samples = self.train_dataset.cardinality().numpy()
        num_train_batches = num_train_samples // self.batch_size

        epochs = 10000
        shuffle_buffer_size = len(self.train_dataset)

        for epoch in tqdm(range(1, epochs + 1)):

            # flat_params = jax.tree_util.tree_leaves(self.state.params)
            # print([np.array(p).shape for p in flat_params])

            train_dataset = self.train_dataset.shuffle(shuffle_buffer_size).batch(self.batch_size)
            # train_dataset = self.train_dataset.batch(self.batch_size)

            train_batch_metrics = []
            train_datagen = iter(tfds.as_numpy(train_dataset))
            for batch_idx in range(num_train_batches):
                batch = next(train_datagen)
                feature_batch, gt_batch = batch
                self.state, metrics, pred = train_step(self.state, batch)
                if epoch % 1000 == 1000-1 and batch_idx == 0:
                    flat_params = jax.tree_util.tree_leaves(self.state.params)
                    flat_params = [np.array(p).ravel() for p in flat_params]
                    flat_params = np.concatenate(flat_params)
                    mean_weight = np.mean(np.abs(flat_params))
                    print("mean_weight", mean_weight)
                    pred_np = np.array(pred)
                    # print("pred", pred_np)
                    # print("gt", gt_batch)
                    print("")
                    if False:
                        plt.figure()
                        plt.scatter(gt_batch, pred_np, marker='.')
                        plt.grid()
                        plt.show()
                train_batch_metrics.append(metrics)
            train_batch_metrics = accumulate_metrics(train_batch_metrics)

            if epoch % 100 == 0:
                metrics = {k: np.array(v).item() for k, v in train_batch_metrics.items()}
                print(metrics)

        print("Training done")



def main():

    dataset = load_or_create_dataset()

    latencies = [r['target'] for r in dataset['dataset']]
    features = np.array([r['features'] for r in dataset['dataset']])
    if False:
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

    trainer = LatencyModelTrainer(dataset)
    trainer.train()

    print("Done")


if __name__ == "__main__":
    main()
