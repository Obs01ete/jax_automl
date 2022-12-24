import numpy as np
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

