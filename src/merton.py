import numpy as np
from numpy.typing import ArrayLike
from math import log, sqrt, pi
from scipy.stats import poisson


lam = 1
m = 10
n = 10

size = (m, n)

x = np.random.poisson(lam, size)