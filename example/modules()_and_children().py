import torch.nn as nn
import torch

print(list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).modules()))
print()
print(list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).children()))
print()

model = list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).modules())[0]
x = torch.FloatTensor([1,2,3,4,5,6,7,8,9,10])
print(model(x))


'''
[Sequential(
  (0): Linear(in_features=10, out_features=20, bias=True)
  (1): ReLU()
), Linear(in_features=10, out_features=20, bias=True), ReLU()]

[Linear(in_features=10, out_features=20, bias=True), ReLU()]

tensor([4.8660, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 8.5535, 5.6076, 0.0000,
        1.7326, 0.0000, 0.0000, 0.0000, 0.0000, 0.2418, 0.0000, 0.0000, 1.5061,
        1.9813, 4.4054], grad_fn=<ThresholdBackward0>)
'''
