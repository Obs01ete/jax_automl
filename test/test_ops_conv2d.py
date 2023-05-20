import unittest

import jax.numpy as jnp

from main import load_or_create_dataset
from latency_model import LatencyModelTrainer
from gradient_automl_conv2d import (
    predict_latencies, rem_loss_fn,
    latency_loss_fn, create_constraints,
    latency_loss_jit_fn, lat_grad_jit_fn, rem_loss_jit_fn,
    rem_grad_fn, rem_grad_jit_fn, cp_value_and_jacobian)


class TestOpsConv2d(unittest.TestCase):

    def test_ops(self):
        op_type = 'conv2d'
        dataset = load_or_create_dataset(op_type)
        trainer = LatencyModelTrainer(dataset, op_type)
        trainer.load_or_train()

        evaluator = trainer.get_evaluator()

        params = evaluator['params']
        predict_flax = evaluator['predict_flax']

        constraints = create_constraints()

        lat_loss, lat_aux_dict = latency_loss_fn(predict_flax, params, variables)
        latency_loss_jit_fn = nn.jit(latency_loss_fn)
        lat_loss, lat_aux_dict = latency_loss_jit_fn(
            predict_flax, params, variables)
        lat_grad_fn = nn.jit(jax.value_and_grad(latency_loss_jit_fn,
                                                argnums=(2, 3),
                                                has_aux=True))
        llg, lat_aux_dict = lat_grad_fn(predict_flax, params, variables)

        raw_latencies = predict_latencies(predict_flax, params, variables)

        tlv = rem_loss_fn(variables)
        
        rem_loss_jit_fn = jax.jit(rem_loss_fn)
        tljv = rem_loss_jit_fn(variables)

        rem_grad_fn = jax.value_and_grad(rem_loss_jit_fn,
                                        argnums=(0,),
                                        has_aux=True)
        gfnv = rem_grad_fn(variables)

        rem_grad_fn_jit = jax.jit(rem_grad_fn)
        gfnjv = rem_grad_fn_jit(variables)

        print("Done")


if __name__ == "__main__":
    unittest.main()
