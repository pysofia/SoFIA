import numpy as np
import numpy.polynomial.hermite_e as H
from scipy.stats import norm
import matplotlib.pyplot as plt

def Herm(n):
    coeffs = [0]*(n + 1)
    coeffs[n] = 1
    return coeffs

def inner_product(h1,h2):
    return lambda x: H.hermeval(x, H.hermemul(h1, h2))

def trapezoid_int(f, a, b, n=100):
    P = [a + i * (b - a) / n for i in range(0,(n + 1))]
    F = [1/2*np.abs(P[i+1]-P[i]) * (f(P[i+1])+f(P[i])) for i in range(0,n)]
    return sum(F)

def unif_icdf(params):
    a = params[0]
    b = params[1]
    return lambda u: u * (b - a) + a

def expo_icdf(params):
    return lambda u: -np.log(1-u)

def norm_icdf(params):
    return lambda u: norm.ppf(u, loc=0, scale=1)

def approximate_rv_coeffs(P, h):
    # Initialize lists for output to make syntax more canonical with the math
    ki = [0]*P

    # Set-up Gauss-Hermite quadrature
    m = P**2
    x, w = H.hermegauss(m)

    # Compute the coefficients, and also build out k in the same pass
    for i in range(0,P):
        # compute the inner product with Gauss-Hermite quadrature
        ip = sum([inner_product(Herm(i),Herm(i))(x[idx])*w[idx] for idx in range(m)])

        #compute the integral
        integrand = lambda u: h(u) * H.hermeval(norm.ppf(u, loc=0, scale=1),Herm(i))
        ki[i] = np.sqrt(2*np.pi) / ip * trapezoid_int(integrand, 0.001, 1-0.001, 1000)

    return ki

def generate_rv(ki, S):
    # build out k termwise
    k = [0]*len(S)
    for i in range(len(ki)):
        k = np.add(k, ki[i] * H.hermeval(S, Herm(i)))
    return k