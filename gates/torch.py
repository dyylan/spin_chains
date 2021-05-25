import torch
import torch.optim as optim
from torch_toffoli_matrices import (toff_a,
                                    toff_b,
                                    toff_c,
                                    toff_d,
                                    toff_e,
                                    toff_f,
                                    toff_g,
                                    toff_h,
                                    toff_i)

n_epochs = 100

dtype = torch.float
dtypec = torch.cfloat

jzz_12 = torch.Tensor([1], requires_grad=True, dtype=dtype)
jzz_23 = torch.Tensor([1], requires_grad=True, dtype=dtype)
jzz_13 = torch.Tensor([1], requires_grad=True, dtype=dtype)
jyy_12 = torch.Tensor([1], requires_grad=True, dtype=dtype)
jxx_23 = torch.Tensor([1], requires_grad=True, dtype=dtype)
jxx_13 = torch.Tensor([1], requires_grad=True, dtype=dtype)
hz_1 = torch.Tensor([1], requires_grad=True, dtype=dtype)
hz_2 = torch.Tensor([1], requires_grad=True, dtype=dtype)
hx_3 = torch.Tensor([1], requires_grad=True, dtype=dtype)

optimizer = optim.SGD([jzz_12, jzz_23, jzz_13, jyy_12, jxx_23, jxx_13, hz_1, hz_2, hx_3], lr=lr)

for epoch in range(n_epochs):
    toffoli_3 = (jzz_12 * toff_a + jzz_23 * toff_b + jzz_13 * toff_c 
                + jyy_12 * toff_d + jxx_23 * toff_e + jxx_13 * toff_f 
                + hz_1 * toff_g + hz_2 * toff_h + hx_3 * toff_i)

    toffoli_3 = 