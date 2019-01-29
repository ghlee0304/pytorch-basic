import torch

x = torch.FloatTensor([1,2,3,4,5,6,7,8,9,10])
print(torch.chunk(x, 3, dim=0)) #10/3 = 3.333인데 올림이 되어 4개 4개 2개로 나뉜다.
print(torch.chunk(x, 5, dim=0))

'''
(tensor([1., 2., 3., 4.]), tensor([5., 6., 7., 8.]), tensor([ 9., 10.]))
(tensor([1., 2.]), tensor([3., 4.]), tensor([5., 6.]), tensor([7., 8.]), tensor([ 9., 10.]))
'''
