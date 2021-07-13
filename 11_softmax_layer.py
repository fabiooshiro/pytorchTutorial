import torch
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
sm = softmax(x)
print('softmax', sm)

ex = np.exp(x)
print(ex)
su = np.sum(np.exp(x))
print(su)
re = ex/su

for indx in range(x.size):
    print(f'{indx}: {re[indx]:.4f}')

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)