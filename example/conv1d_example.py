import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1,1,3)

    def forward(self, x):
        result = self.conv1(x)
        return result

x = torch.FloatTensor([[[1,2,3,4,5,6,7,8,9,10]]])
m = Model()
a = m.forward(x)
print(a)

'''
tensor([[[-1.4666, -2.1935, -2.9204, -3.6472, -4.3741, -5.1010, -5.8279,
          -6.5548]]], grad_fn=<SqueezeBackward1>)
'''
