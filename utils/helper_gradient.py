import torch

def flat_grad(y, x, retain_graph = False, create_graph = False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()

def kl_div_categorical(mu_old, mu_new, eps=1e-8):
    mu_old = mu_old + eps
    mu_new = mu_new + eps

    #Probably uncessary as softmax normalizes already
    mu_old = mu_old / mu_old.sum(dim = -1, keepdim = True)
    mu_new = mu_new / mu_new.sum(dim=-1, keepdim = True)

    kl = torch.sum(mu_old *(torch.log(mu_old) - torch.log(mu_new)), dim=-1).mean()
    return kl



def conjugate_gradient(A, b, delta = 1e-10, max_iterations = 10):
    #A*x = b return using conjugate gradient method
    #Paper recommended max_iters = 10
    x = torch.zeros_like(b)
    r = b.clone() #residuals
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p


        if (x - x_new).norm() <= delta:
            return x_new

        i += 1

        r = r - alpha * AVP

        beta = (r @ r) / dot_old

        p = r + beta*p

        x = x_new

    return x





