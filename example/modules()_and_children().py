import torch.nn as nn
import torch


print(list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).modules()))
print()
print(list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).children()))
print()

model = list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).modules())[0]
x = torch.FloatTensor([1,2,3,4,5,6,7,8,9,10])
print(model(x))

print()
print(list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).named_modules()))
print()
print(list(nn.Sequential(nn.Linear(10, 20), nn.ReLU()).named_children()))
print()
# 결과에서 '0'하고 '1'이 이름인것 같음

'''
[Sequential(
  (0): Linear(in_features=10, out_features=20, bias=True)
  (1): ReLU()
), Linear(in_features=10, out_features=20, bias=True), ReLU()]

[Linear(in_features=10, out_features=20, bias=True), ReLU()]

tensor([2.1059, 3.6117, 0.0000, 6.4481, 4.1916, 0.0000, 5.1995, 0.0000, 7.1257,
        0.8863, 0.0000, 0.0000, 0.5251, 0.0000, 0.0000, 0.0000, 2.5678, 2.7131,
        4.2904, 7.9817], grad_fn=<ThresholdBackward0>)

[('', Sequential(
  (0): Linear(in_features=10, out_features=20, bias=True)
  (1): ReLU()
)), ('0', Linear(in_features=10, out_features=20, bias=True)), ('1', ReLU())]

[('0', Linear(in_features=10, out_features=20, bias=True)), ('1', ReLU())]
'''
