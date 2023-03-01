import sys
sys.path.append('../')

import numpy as np
from util.metrics import MSE
from util.metrics import METRICS

I = np.random.rand(4, 4, 3)*255
I = I.astype(np.uint8)

assert(MSE(I, I+2) == 4)

