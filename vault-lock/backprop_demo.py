"""
Backpropagation and gradient descent with no autograd: the maths train.py
leaves to loss.backward(), done by hand for a small network, then checked
against PyTorch.

    python backprop_demo.py

Part 1: a 2-layer net (input 4 -> hidden 5 -> 3 classes), one forward pass,
        the chain rule backwards by hand, compared with loss.backward().
Part 2: the same for a conv layer and a 2x2 max pool: the two layers a CNN
        adds, and what their gradients do.
Part 3: gradient descent by hand on that 2-layer net: the loss going down.
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)


def same(name, mine, torch_says):
    err = (mine - torch_says).abs().max().item()
    print(f"  {name:10s} by hand vs autograd: largest difference {err:.1e}  {'OK' if err < 1e-5 else 'WRONG'}")


# ------------------------------------------------------------------------------
print("Part 1: backprop through a 2-layer net, by hand")
# x (batch 8, 4 inputs) -> z1 = x W1^T + b1 -> h = ReLU(z1) -> z2 = h W2^T + b2
# -> p = softmax(z2) -> loss = mean(-log p[right class])
#
# WIDTH of this net = 5 hidden units; DEPTH = 2 layers with weights.
x = torch.randn(8, 4)
y = torch.randint(0, 3, (8,))
W1 = torch.randn(5, 4, requires_grad=True)
b1 = torch.zeros(5, requires_grad=True)
W2 = torch.randn(3, 5, requires_grad=True)
b2 = torch.zeros(3, requires_grad=True)

# Forward, keeping what the backward pass needs.
z1 = x @ W1.T + b1
h = z1.clamp(min=0)
z2 = h @ W2.T + b2
loss = F.cross_entropy(z2, y)
loss.backward()                         # autograd's answer, to check against

with torch.no_grad():
    n = x.shape[0]
    p = torch.softmax(z2, dim=1)
    onehot = F.one_hot(y, 3).float()
    # d loss / d z2: softmax + cross-entropy together have the neat derivative
    # (p - onehot), divided by n because the loss is a mean over the batch.
    dz2 = (p - onehot) / n
    # z2 = h W2^T + b2: each weight's gradient = (gradient arriving at its
    # output) x (the input it multiplied), summed over the batch.
    dW2 = dz2.T @ h
    db2 = dz2.sum(0)
    # ...and the gradient passed back to h goes back through the same weights.
    dh = dz2 @ W2
    # ReLU passes the gradient where it was 'on' (z1 > 0) and blocks it elsewhere.
    dz1 = dh * (z1 > 0).float()
    dW1 = dz1.T @ x
    db1 = dz1.sum(0)

same("dL/dW2", dW2, W2.grad)
same("dL/db2", db2, b2.grad)
same("dL/dW1", dW1, W1.grad)
same("dL/db1", db1, b1.grad)


# ------------------------------------------------------------------------------
print("\nPart 2: backprop through a 3x3 convolution and a 2x2 max pool")
img = torch.randn(1, 1, 6, 6)
K = torch.randn(1, 1, 3, 3, requires_grad=True)
out = F.conv2d(img, K)                   # (1, 1, 4, 4): the kernel slid over the image
pooled = F.max_pool2d(out, 2)            # (1, 1, 2, 2): biggest of each 2x2 block
target = torch.randn(1, 1, 2, 2)
loss = ((pooled - target) ** 2).sum()
loss.backward()

with torch.no_grad():
    d_pooled = 2 * (pooled - target)
    # Max pool: only the biggest value in each block affected the output, so
    # the whole gradient goes to it; the other 3 get 0.
    d_out = torch.zeros_like(out)
    for i in range(2):
        for j in range(2):
            block = out[0, 0, 2 * i:2 * i + 2, 2 * j:2 * j + 2]
            r, c = divmod(block.argmax().item(), 2)
            d_out[0, 0, 2 * i + r, 2 * j + c] = d_pooled[0, 0, i, j]
    # Conv: the SAME 9 kernel weights were used at every position, so each
    # weight's gradient is a sum over all positions of
    # (gradient at that output) x (the pixel under that weight).
    dK = torch.zeros(3, 3)
    for i in range(4):
        for j in range(4):
            dK += d_out[0, 0, i, j] * img[0, 0, i:i + 3, j:j + 3]

same("dL/dkernel", dK, K.grad[0, 0])
print("  (weight sharing: 9 weights, but their gradient collects from all 16 positions)")


# ------------------------------------------------------------------------------
print("\nPart 3: gradient descent by hand on the Part 1 net")
# A toy problem: 3 classes of points in 4-D, around different centres.
centres = torch.randn(3, 4) * 2
y = torch.randint(0, 3, (120,))
x = centres[y] + torch.randn(120, 4)
params = [torch.randn(5, 4) * 0.5, torch.zeros(5), torch.randn(3, 5) * 0.5, torch.zeros(3)]
lr = 0.5
for step in range(201):
    W1, b1, W2, b2 = params
    z1 = x @ W1.T + b1
    h = z1.clamp(min=0)
    z2 = h @ W2.T + b2
    p = torch.softmax(z2, dim=1)
    loss = -torch.log(p[torch.arange(len(y)), y]).mean()
    # backward (Part 1's equations)
    dz2 = (p - F.one_hot(y, 3).float()) / len(y)
    dz1 = (dz2 @ W2) * (z1 > 0).float()
    grads = [dz1.T @ x, dz1.sum(0), dz2.T @ h, dz2.sum(0)]
    # gradient descent: every weight takes a small step downhill
    params = [w - lr * g for w, g in zip(params, grads)]
    if step % 40 == 0:
        acc = (z2.argmax(1) == y).float().mean().item()
        print(f"  step {step:3d}: loss {loss.item():.3f}  accuracy {acc:.0%}")
print("  The loss goes down because each step moves the weights against the gradient,")
print("  the direction in which the loss increases fastest.")
