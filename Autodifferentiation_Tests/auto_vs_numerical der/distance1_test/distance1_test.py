import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD


plumedInit = {
    "COMPONENTS": {
        "Distance": {"period": None, "derivative": True},}}


def scrach_distance(positions):
    x1,y1,z1 = positions[0]
    x2,y2,z2 = positions[1]
    dist = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
    return dist

grad_distance = grad(scrach_distance)
dummy_box = jnp.zeros((3,3))

def distance_fn(action: PLMD.PythonCVInterface):
    positions= action.getPositions()
    d = scrach_distance(positions)
    grad_d = grad_distance(positions)
    # print(positions)
    return d,grad_d,dummy_box
