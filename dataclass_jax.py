import jax
import dataclasses


def register_pytree_node_dataclass(cls):
    _flatten = lambda obj: jax.tree_util.tree_flatten(dataclasses.asdict(obj))
    _unflatten = lambda d, children: cls(**d.unflatten(children))
    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls
