import torch


torch.random.manual_seed(0)
batch1 = torch.ones(1, 2, 2)
batch2 = torch.ones(1, 2, 4)
res = torch.bmm(batch1, batch2)


print()
print(batch1)
print()
print(batch2)
print()
print(res)
print()


'''
tensor([[[1., 1.],
         [1., 1.]]])

tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.]]])

tensor([[[2., 2., 2., 2.],
         [2., 2., 2., 2.]]])
'''
