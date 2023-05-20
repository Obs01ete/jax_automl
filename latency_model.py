import os
import math
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)
import optax


class LatencyNet(nn.Module):

    @nn.compact
    def __call__(self, x):
        ch = 128

        linear1 = nn.Dense(features=ch)
        linear2 = nn.Dense(features=ch)
        linear3 = nn.Dense(features=ch)
        linear4 = nn.Dense(features=ch)
        linear5 = nn.Dense(features=ch)
        linear6 = nn.Dense(features=ch)
        linear7 = nn.Dense(features=ch)
        linear8 = nn.Dense(features=ch)
        linear9 = nn.Dense(features=1)

        x = 1e-3 * x
        x = nn.relu(linear1(x))
        x = nn.relu(linear2(x))
        x = nn.relu(linear3(x))
        x = nn.relu(linear4(x))
        x = nn.relu(linear5(x))
        x = nn.relu(linear6(x))
        x = nn.relu(linear7(x))
        x = nn.relu(linear8(x))
        x = linear9(x)
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
        apply_fn=model.apply,
        tx=optimizer,
        params=variables['params']
    )


def total_loss(*, pred, label):
    smoothener_sec = 0.001
    total = mape_metric(pred=pred, label=label, smoothener=smoothener_sec)
    return total


def mape_metric(*, pred, label, smoothener=0.0):
    losses = jnp.abs(pred - label) / (label + smoothener)
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


@jax.jit
def evaluate(
    state: train_state.TrainState, batch: jnp.ndarray
):
    feature, label = batch
    pred = state.apply_fn({'params': state.params}, feature)
    metrics = compute_metrics(pred=pred, label=label)
    return metrics, pred


@partial(jax.jit, static_argnums=(0,))
def predict(
    model, params, feature: jnp.ndarray
):
    pred = model.apply({'params': params}, feature)
    return pred


predict_flax = nn.jit(LatencyNet)


class LatencyModelTrainer:
    def __init__(self, dataset, name='linear'):
        self.dataset = dataset
        self.features = np.array([r['features'] for r in dataset['dataset']])
        self.targets = np.array([r['target'] for r in dataset['dataset']])
        self.name = name

        self.batch_size = 128
        learning_rate = 1e-5

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.features, self.targets))
        # lin_feat = np.tile(np.expand_dims(np.linspace(0.01/1000, 0.01, 1000), 1), (1, 2))
        # lin_targets = np.linspace(0.01/1000, 0.01, 1000)
        # self.train_dataset = tf.data.Dataset.from_tensor_slices((lin_feat, lin_targets))

        self.net = LatencyNet()

        self.rng = jax.random.PRNGKey(43)
        self.state = init_train_state(
           self.net, self.rng, (self.batch_size, self.features.shape[1]),
           learning_rate)

        # feature = jnp.ones((1, 2))
        # pred = predict(self.net, self.state.params, feature)

        # pred_flax = predict_flax().bind({'params': self.state.params})(feature)

        self.checkpoint_dir = f"checkpoint_{self.name}"
        print("")

    @staticmethod        
    def print_metrics(metrics):
        metrics = {k: np.array(v).item() for k, v in metrics.items()}
        print(metrics)

    def train(self):
        num_train_samples = self.train_dataset.cardinality().numpy()
        num_train_batches = num_train_samples // self.batch_size

        epochs = math.ceil(2_000_000 / num_train_samples)
        shuffle_buffer_size = len(self.train_dataset)

        for epoch in tqdm(range(1, epochs + 1)):

            # flat_params = jax.tree_util.tree_leaves(self.state.params)
            # print([np.array(p).shape for p in flat_params])

            train_dataset = self.train_dataset.shuffle(shuffle_buffer_size) \
                .batch(self.batch_size)
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
                self.print_metrics(train_batch_metrics)

        self.print_metrics(train_batch_metrics)

        self.save()

        print("Training done")

    def evaluate(self):
        train_batch_metrics = []
        data_iterator = iter(tfds.as_numpy(
            self.train_dataset.batch(self.batch_size)))
        preds = []
        gts = []
        for batch in data_iterator:
            features, gt = batch
            metrics, pred = evaluate(self.state, batch)
            train_batch_metrics.append(metrics)
            preds.extend([v.item() for v in list(pred)])
            gts.extend([v.item() for v in list(gt)])
        train_batch_metrics = accumulate_metrics(train_batch_metrics)
        self.print_metrics(train_batch_metrics)

        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(gts, preds, marker='.')
            plt.grid()
            plt.show()

    def save(self):
        checkpoints.save_checkpoint(ckpt_dir=self.checkpoint_dir,
                                    target=self.state, step=0, overwrite=True)
        print(f"Saved to {self.checkpoint_dir}")

    def load_or_train(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, "checkpoint_0")):
            restored_state = checkpoints.restore_checkpoint(
                ckpt_dir=self.checkpoint_dir, target=self.state)
            # assert jax.tree_util.tree_all(jax.tree_map(
            #     lambda x, y: (x == y).all(), self.state.params, restored_state.params))
            self.state = restored_state
        else:
            self.train()

    def get_evaluator(self):
        return dict(
            predict_fn=predict,
            module=self.net,
            params=self.state.params,
            predict_flax=predict_flax(),
            )
