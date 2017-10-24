import numpy as np
import pylab as plt
from Slice_Sampler import *


start = timeit.default_timer()

#Your statements here
x=run(100000)
stop = timeit.default_timer()

print stop - start
n, bins, patches = plt.hist(x, 50, normed=1, histtype='step', lw=2, color='r', label="Beta")
plt.show()

