import torch

x = torch.FloatTensor([1,2,3,4,5,6,7,8,9,10])
print(torch.chunk(x, 3, dim=0))
print(torch.chunk(x, 5, dim=0))

'''
(tensor([1., 2., 3., 4.]), tensor([5., 6., 7., 8.]), tensor([ 9., 10.]))
(tensor([1., 2.]), tensor([3., 4.]), tensor([5., 6.]), tensor([7., 8.]), tensor([ 9., 10.]))
'''
