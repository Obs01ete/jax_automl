import unittest

import jax.numpy as jnp

from main import load_or_create_dataset
from latency_model import LatencyModelTrainer
from gradient_automl_linear import (
    predict_latencies, rem_loss_fn,
    latency_loss_fn, create_constraints,
    latency_loss_jit_fn, lat_grad_jit_fn, rem_loss_jit_fn,
    rem_grad_fn, rem_grad_jit_fn, cp_value_and_jacobian)


class TestOpsLinear(unittest.TestCase):

    def test_ops(self):
        op_type = 'linear'
        dataset = load_or_create_dataset(op_type)
        trainer = LatencyModelTrainer(dataset, op_type)
        trainer.load_or_train()

        evaluator = trainer.get_evaluator()

        params = evaluator['params']
        predict_flax = evaluator['predict_flax']

        constraints = create_constraints()

        input_features_size = 10

        test_features_array = jnp.zeros((constraints.layers.max,),
                                        dtype=jnp.float32) + 512
        test_features_array = test_features_array.at[5].set(-1000)

        lat_args = (predict_flax, params,
                    input_features_size,
                    test_features_array,
                    constraints)
        lat_loss, lat_aux_dict = latency_loss_fn(*lat_args)
        lat_loss, lat_aux_dict = latency_loss_jit_fn(*lat_args)
        llg, lat_aux_dict = lat_grad_jit_fn(*lat_args)

        raw_latencies = predict_latencies(predict_flax, params,
                                          input_features_size,
                                          test_features_array)

        rem_args = (input_features_size, test_features_array, constraints)
        tlv = rem_loss_fn(*rem_args)
        tljv = rem_loss_jit_fn(*rem_args)
        gfnv = rem_grad_fn(*rem_args)
        gfnjv = rem_grad_jit_fn(*rem_args)

        vnj = cp_value_and_jacobian(test_features_array,
                                    predict_flax,
                                    params,
                                    input_features_size,
                                    constraints)

        print("Done")


if __name__ == "__main__":
    unittest.main()
