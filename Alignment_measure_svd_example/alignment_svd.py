import jax.numpy as jnp
from jax import grad, jit
import plumedCommunications as PLMD


plumedInit = {
    "COMPONENTS": {
        "Alighnemt": {"period": None, "derivative": True},}}

@jit
def B_circumcenter(A, B, C):
   #link 
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
   
    centers = jnp.zeros((len(positions)-2, 3))
    # distance_pace = jnp.zeros((len(positions)-2, 1))
    for i in range(len(positions) - 2):
        A, B, C = positions[i, :], positions[i+1, :], positions[i+2, :]
        cc = B_circumcenter(A, B, C)
        # jax.debug.print("{cc}", cc=cc)   
        centers = centers.at[i, :].set(cc)
        
    return  centers


def svd_fit(points):

    avg = jnp.mean(points, axis=0)  
    sigma = jnp.std(points, axis=0)
    subtracted = (points - avg)/sigma

    _, s, _ = jnp.linalg.svd(subtracted, full_matrices=False) 
    
    s1 = s[0]
    s_sum = jnp.sum(s)

    return s1/s_sum

grad_fit = grad(svd_fit)


def alignment(action: PLMD.PythonCVInterface):
    x= action.getPositions()
    c_points = circumcenter_matrix(x)
    straightness = svd_fit(c_points)
    straightness_grad = grad_fit(x)

    dummy_grad_box = jnp.zeros((3,3))
    
    return straightness , straightness_grad, dummy_grad_box
    
