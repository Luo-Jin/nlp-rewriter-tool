import torch
import random

a = torch.rand(9)
print('a:\n', a)

random.shuffle(a)
print('random.shuffle(a):\n', a)

index = [i for i in range(len(a))]
print('index\n', index)
random.shuffle()
print('random.shuffle(index):\n', index)
print('shuffle tensor:\n', a[index])

