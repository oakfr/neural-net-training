# self-attention:
# each token emits two vectors: a query vector, and a key vector
# query = what am i looking for
# key = what do i contain
# affinities between tokens is a dot products between the keys and the queries
# finally, when we aggregate the tokens, we don't aggregate tokens, but the "values"
# query, key, value => here is what am i interested in, here is what i have,
# and if you find me interesting, here is what i will communicate to you
#

import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)
B,T,C = 4,8,32
x = torch.randn(B,T,C)

# let's see a single head perform attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)   # B, T, 16
q = query(x) # B, T, 16

wei = q @ k.transpose(-2, -1)  # B, T, T

tril = torch.tril(torch.ones(T,T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei @ v

print(out.shape)

print(tril)

print(wei)

