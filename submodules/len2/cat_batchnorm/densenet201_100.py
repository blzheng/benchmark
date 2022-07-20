import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.batchnorm2d198 = BatchNorm2d(1888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565, x572, x579, x586, x593, x600, x607, x614, x621, x628, x635, x642, x649, x656, x663, x670, x677, x684, x691, x698):
        x699=torch.cat([x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565, x572, x579, x586, x593, x600, x607, x614, x621, x628, x635, x642, x649, x656, x663, x670, x677, x684, x691, x698], 1)
        x700=self.batchnorm2d198(x699)
        return x700

m = M().eval()
x481 = torch.randn(torch.Size([1, 896, 7, 7]))
x488 = torch.randn(torch.Size([1, 32, 7, 7]))
x495 = torch.randn(torch.Size([1, 32, 7, 7]))
x502 = torch.randn(torch.Size([1, 32, 7, 7]))
x509 = torch.randn(torch.Size([1, 32, 7, 7]))
x516 = torch.randn(torch.Size([1, 32, 7, 7]))
x523 = torch.randn(torch.Size([1, 32, 7, 7]))
x530 = torch.randn(torch.Size([1, 32, 7, 7]))
x537 = torch.randn(torch.Size([1, 32, 7, 7]))
x544 = torch.randn(torch.Size([1, 32, 7, 7]))
x551 = torch.randn(torch.Size([1, 32, 7, 7]))
x558 = torch.randn(torch.Size([1, 32, 7, 7]))
x565 = torch.randn(torch.Size([1, 32, 7, 7]))
x572 = torch.randn(torch.Size([1, 32, 7, 7]))
x579 = torch.randn(torch.Size([1, 32, 7, 7]))
x586 = torch.randn(torch.Size([1, 32, 7, 7]))
x593 = torch.randn(torch.Size([1, 32, 7, 7]))
x600 = torch.randn(torch.Size([1, 32, 7, 7]))
x607 = torch.randn(torch.Size([1, 32, 7, 7]))
x614 = torch.randn(torch.Size([1, 32, 7, 7]))
x621 = torch.randn(torch.Size([1, 32, 7, 7]))
x628 = torch.randn(torch.Size([1, 32, 7, 7]))
x635 = torch.randn(torch.Size([1, 32, 7, 7]))
x642 = torch.randn(torch.Size([1, 32, 7, 7]))
x649 = torch.randn(torch.Size([1, 32, 7, 7]))
x656 = torch.randn(torch.Size([1, 32, 7, 7]))
x663 = torch.randn(torch.Size([1, 32, 7, 7]))
x670 = torch.randn(torch.Size([1, 32, 7, 7]))
x677 = torch.randn(torch.Size([1, 32, 7, 7]))
x684 = torch.randn(torch.Size([1, 32, 7, 7]))
x691 = torch.randn(torch.Size([1, 32, 7, 7]))
x698 = torch.randn(torch.Size([1, 32, 7, 7]))
start = time.time()
output = m(x481, x488, x495, x502, x509, x516, x523, x530, x537, x544, x551, x558, x565, x572, x579, x586, x593, x600, x607, x614, x621, x628, x635, x642, x649, x656, x663, x670, x677, x684, x691, x698)
end = time.time()
print(end-start)
