# Basic Architecture
# y = mx + c
import numpy as np
# Steps:
#   Initialise Parameters
x = np.random.randn(10,1)
y = 5*x + np.random.rand()
m = 0.0
c = 0.0
# Hyperparameter
rate_of_learning = 0.01

# main function
def main(x, y, m, c, rate_of_learning):
    dl_dm = 0.0
    dl_dc = 0.0
    N = x.shape[0]
    # loss = (y-(mx+c)))^2
    for xi, yi in zip(x,y):
        # Chain Rule
        dl_dm = -2*x*(yi-(m*xi+c))  
        dl_dc = -2*xi*(yi-(m*xi+c))
    m = m - rate_of_learning*(1/N)*dl_dm
    c = c - rate_of_learning*(1/N)*dl_dc
    
    return m, c


#   Updates
for epoch in range(400):
    w, c = main(x, y, m, c, rate_of_learning)
    print(w,c)
    Δy = m*x + c
    loss = np.divide(np.sum((y-Δy)**2, axis=0), x.shape[0]) 
    print(f'{epoch} loss is {loss}, parameters m:{m}, c:{c}')
    # Run Gradient Descent Algorithm
    pass
