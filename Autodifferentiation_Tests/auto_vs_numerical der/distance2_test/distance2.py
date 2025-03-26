import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD


plumedInit = {
    "COMPONENTS": {
        "Distance": {"period": None, "derivative": True},}}

@jit
def distance(positions):
    dist = jnp.linalg.norm(positions[0]-positions[1])
    return dist

grad_distance = grad(distance)
dummy_box = jnp.zeros((3,3))

def distance_fn(action: PLMD.PythonCVInterface):
    positions= action.getPositions()
    d = distance(positions)
    grad_d = grad_distance(positions)
    # print(positions)
    return d,grad_d,dummy_box
    

    