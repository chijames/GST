"""Basic or helper implementation."""

import torch
from torch import nn
from torch.nn import functional

def convert_to_one_hot(indices, num_classes):
    """
    Args:
        indices (tensor): A vector containing indices,
            whose size is (batch_size,).
        num_classes (tensor): The number of classes, which would be
            the second dimension of the resulting one-hot matrix.

    Returns:
        result: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1)
    return one_hot


def masked_softmax(logits, mask=None, dim=-1):
    eps = 1e-20
    probs = functional.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        ddif = len(probs.shape) - len(mask.shape)
        mask = mask.view(mask.shape + (1,)*ddif) if ddif>0 else mask
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)
    return probs


def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1],
                                 num_classes=logits.size(1))
    return one_hot

def gst_mover(logits, temperature=1.0, mask=None, hard=True, gap=1.0, detach=True):
    logits_cpy = logits.detach() if detach else logits
    probs = masked_softmax(logits_cpy, mask)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs = probs)  
    action = m.sample() 
    argmax = probs.argmax(dim=-1, keepdim=True)
    
    action_bool = action.bool()
    max_logits = torch.gather(logits_cpy, -1, argmax)
    move = (max_logits - logits_cpy)*action
    
    if type(gap)!=float:
        pi = probs[action_bool]*(1-1e-5) # for numerical stability
        gap = ( -(1-pi).log()/pi ).view( logits.shape[:-1] + (1,) )
    move2 = ( logits_cpy + (-max_logits + gap) ).clamp(min=0.0)
    move2[action_bool] = 0.0
    logits = logits + (move - move2)

    logits = logits - logits.mean(dim=-1, keepdim=True)
    prob = masked_softmax(logits=logits / temperature, mask=mask)
    action = action - prob.detach() + prob if hard else prob
    return action.reshape(logits.shape)

def rao_gumbel(logits, temperature=1.0, mask = None, repeats=100, hard=True):
    logits_cpy = logits.detach()
    probs = masked_softmax(logits_cpy, mask)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs = probs)  
    action = m.sample()
    action_bool = action.bool()
        
    logits_shape = logits.shape
    bs = logits_shape[0]
        
    E = torch.empty(logits_shape + (repeats,), 
                    dtype=logits.dtype, 
                    layout=logits.layout, 
                    device=logits.device,
                    memory_format=torch.legacy_contiguous_format).exponential_()
                
    Ei = E[action_bool].view(logits_shape[:-1] + (repeats,)) # rv. for the sampled location 

    wei = logits_cpy.exp()
    if mask is not None:
        wei = wei * mask.float() + 1e-20
    Z = wei.sum(dim=-1, keepdim=True) # (bs, latdim, 1)
    EiZ = (Ei/Z).unsqueeze(-2) # (bs, latdim, 1, repeats)

    new_logits = E / (wei.unsqueeze(-1)) # (bs, latdim, catdim, repeats)
    new_logits[action_bool] = 0.0
    new_logits = -(new_logits + EiZ + 1e-20).log()
    logits_diff = new_logits - logits_cpy.unsqueeze(-1)

    prob = masked_softmax((logits.unsqueeze(-1) + logits_diff)/temperature, mask, dim=-2).mean(dim=-1)
    action = action - prob.detach() + prob if hard else prob
    return action.view(logits_shape)


def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.

    Args:
        logits (tensor): A un-normalized probability values,
            which has the size (batch_size, num_classes)
        temperature (float): A temperature parameter. The higher
            the value is, the smoother the distribution is.
        mask (tensor, optional): If given, it masks the softmax
            so that indices of '0' mask values are not selected.
            The size is (batch_size, num_classes).

    Returns:
        y: The sampled output, which has the property explained above.
    """
    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y


def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand.to(sequence_length)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (tensor): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A tensor with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length-1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    reversed_indices = reversed_indices.to(inputs).long()
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

###### deprecated #####
'''
from scipy.special import exp1

def exp_int_apx(x):
    upper = torch.log(1 + 1/x)
    lower = torch.log(1+2/x)/2
    wei = 0.11*torch.log(1e-10 + 5/x**0.9) + 0.14
    return upper*wei + lower*(1-wei)

def exp_int(x):
    x_cpu = x.cpu()
    ans = x_cpu.exp() * exp1(x_cpu)
    return ans.to(x.device)

def conmax_mover(logits, temperature=1.0, mask=None, hard=True, 
                 training=True, noi=0.01, apx=False):
    logits_cpy = logits.detach()
    probs = masked_softmax(logits_cpy, mask)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs = probs)  
    action = m.sample()
    argmax = probs.argmax(dim=-1, keepdim=True)
    
    max_probs = torch.gather(probs, -1, argmax)
    max_logits = torch.gather(logits_cpy, -1, argmax)
    if training:
        z = torch.empty_like(max_probs,memory_format=torch.legacy_contiguous_format).uniform_()
        z = (1 - noi) + (noi*2)*z
        max_probs = max_probs*z
        max_logits = max_logits + z.log()
    
    if apx:
        new_logits = -exp_int_apx(probs/max_probs) # the argument is EiPj
    else:
        new_logits = -exp_int(probs/max_probs) # the argument is EiPj
    
    new_logits[action==1.0] = 0.0
    new_logits = new_logits + max_logits
    logits = logits - logits_cpy + new_logits
    
    logits = logits - logits.mean(dim=-1, keepdim=True)
    prob = masked_softmax(logits=logits / temperature, mask=mask)
    
    action = action - prob.detach() + prob if hard else prob
    return action.reshape(logits.shape)
'''
