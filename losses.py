import jax.numpy as jnp


def double_boundary_loss(values,
                         min_value,
                         max_value,
                         min_slope=1.0,
                         max_slope=1.0):
    return jnp.maximum(
        max_slope * jnp.maximum(0, (values - max_value) / max_value),
        min_slope * jnp.maximum(0, (min_value - values) / min_value),
        )
