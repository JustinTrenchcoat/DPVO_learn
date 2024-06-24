import torch
import cuda_corr
# cuda_corr is a package that created through the setup.py under the DPVO folder. 
# Every function that is called through the package is related to the source files that this package is linked to. 
# Please refer to the comments in setup.py to see further.

class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, ii, jj, radius, dropout):
        """ forward correlation """
        # no idea why the triple quotes are here
        # save_for_backward() stores values generated during forwards that will be 
        # needed when doing backward().
        ctx.save_for_backward(fmap1, fmap2, coords, ii, jj)
        ctx.radius = radius
        ctx.dropout = dropout
        #the result of cuda_corr.forward() function is returned to corr.
        corr, = cuda_corr.forward(fmap1, fmap2, coords, ii, jj, radius)

        return corr
    
    @staticmethod
    def backward(ctx,grad):
        """ backward correlation """
        #saved_tensors would return the saved values that were previously saved through save_for_backward() funciton
        fmap1,fmap2,coords,ii,jj=ctx.saved_tensors

        if ctx.dropout <1:
            perm = torch.rand(len(ii), device="cuda") < ctx.dropout
            coords = coords[:,perm]
            grad = grad[:,perm]
            ii = ii[perm]
            jj = jj[perm]
        # the '\' is the continuation character, the line length is restricted so it is broken in two parts
        # and connected by '\'
        fmap1_grad, fmap2_grad = \
            cuda_corr.backward(fmap1,fmap2,coords,ii,jj,grad,ctx.radius)
        
        return fmap1_grad, fmap2_grad,None,None,None,None,None
    
class PatchLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,net,coords,radius):
        """forward patchify"""
        ctx.radius=radius
        ctx.save_for_backward(net,coords)

        patches, = cuda_corr.patchify_forward(net,coords,radius)
        return patches
    @staticmethod
    def backward(ctx, grad):
        """backward patchify"""
        net,coords = ctx.saved_tensors
        grad,=cuda_corr.patchify_backward(net,coords,grad,ctx.radius)

        return grad, None, None
    
def patchify(net, coords, radius, mode='bilinear'):
    """extract patches"""
    patches = PatchLayer.apply(net,coords,radius)

    if mode == 'bilinear':
        offset = (coords - coords.floor()).to(net.device)
        dx,dy = offset[:,:,None,None,None].unbind(dim=-1)

        d = 2*radius + 1
        x00 = (1-dy)*(1-dx)*patches[...,:d,:d]
        x01 = (1-dy)*(  dx)*patches[...,:d,1:]
        x10 = (  dy)*(1-dx)*patches[...,1:,:d]
        x11 = (  dy)*(  dx)*patches[...,1:,1:]

        return x00 + x01 + x10 + x11
    
    return patches

def corr(fmap1,fmap2,coords,ii,jj,radius=1,dropout=1):
    return CorrLayer.apply(fmap1,fmap2,coords,ii,jj,radius,dropout)
