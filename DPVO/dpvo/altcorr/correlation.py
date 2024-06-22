import torch
import cuda_corr
#cuda_corr is a package that created through the setup.py under the DPVO folder. Every function that is called through the package is related to the source files that this package is linked to. Please refer to the comments in setup.py to see further.

class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, ii, jj, radius, dropout):
        """ forward correlation """
        #no idea why the triple quotes are here
        ctx.save_for_backward(fmap1, fmap2, coords, ii, jj)
        ctx.radius = radius
        ctx.dropout = dropout
        corr, = cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)

        return corr
