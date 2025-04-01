import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD
import jax
jax.config.update("jax_enable_x64", True) 

plumedInit = {
    "COMPONENTS": {
        "Alighnemt": {"period": None, "derivative": True},}}

@jit
def B_circumcenter(A, B, C):
   #https://math.stackexchange.com/questions/1304194/center-of-circle-when-three-points-in-3-space-are-given/1304202#1304202
    a =jnp.linalg.norm(B - C)
    b =jnp.linalg.norm( C - A)
    c =jnp.linalg.norm( A - B)

    alpha = (a**2)*(b**2 + c**2 - a**2)
    beta = (b**2)*(c**2 + a**2 - b**2)
    gamma = (c**2)*(a**2 + b**2 - c**2)
    
    denomenator = (alpha + beta + gamma)
    center_point = (alpha*A + beta*B + gamma*C )/ denomenator
    return center_point

@jit
def circumcenter_matrix(positions):
    N = positions.shape[0]
    indices = jnp.arange(N - 2)
    
    def compute_cc(i):
        A = positions[i]
        B = positions[i + 1]
        C = positions[i + 2]
        return B_circumcenter(A, B, C)
    
    center = jax.vmap(compute_cc)(indices)
    return center


def svd_fit(points):

    avg = jnp.mean(points, axis=0)  
    sigma = jnp.std(points, axis=0)
    subtracted = (points - avg)/sigma
    _, s, _ = jnp.linalg.svd(subtracted, full_matrices=False) 
    s1 = s[0]
    s_sum = jnp.sum(s)
    return s1/s_sum

grad_fit = grad(svd_fit)


def distance_from_positions(positions: jnp.ndarray):
    cm = circumcenter_matrix(positions)
    return svd_fit(cm)


gradient = jit(grad(distance_from_positions))
dummy_box = jnp.zeros((3,3))

def alignment(action: PLMD.PythonCVInterface):
    
    x = action.getPositions()  
    straightness = distance_from_positions(x)
    grad_d = gradient(x)
    return straightness, grad_d, dummy_box