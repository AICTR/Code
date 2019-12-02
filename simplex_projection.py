import numpy as np

def simplex_projection_selfnorm2(v,b):
    while (max(abs(v)) > 1e6):
        v = v / 10
        break
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    c = np.array(range(1, len(u)+1, 1))
    sample = u - (sv - b) / c
    sample = sample[sample>0]
    rho = np.argmin(sample)
    theta = (sv[rho] - b) / (rho+1)
    w = np.maximum(v - theta, 0)
    return w