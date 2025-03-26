import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD


plumedInit = {
    "COMPONENTS": {
        "Distance": {"period": None, "derivative": True},}}


def helper_fn(positions):
    x,y,z = positions[0]
    fn = 2*x**2 + 2*y**2 + 2*z**2
    return fn

grad_distance = grad(helper_fn)
dummy_box = jnp.zeros((3,3))

def scratch_fn(action: PLMD.PythonCVInterface):
    positions= action.getPositions()
    s = helper_fn(positions)
    
    grad_s = grad_distance(positions)
    # print(positions)
    return s,grad_s,dummy_box
