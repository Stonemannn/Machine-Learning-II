import torch

i1 = 2.
i2 = 3.

i1 = torch.tensor(i1)
i2 = torch.tensor(i2)

print()

w1 = torch.tensor([0.11], requires_grad=True)
w2 = torch.tensor([0.21], requires_grad=True)
w3 = torch.tensor([0.12], requires_grad=True)
w4 = torch.tensor([0.08], requires_grad=True)
w5 = torch.tensor([0.14], requires_grad=True)
w6 = torch.tensor([0.15], requires_grad=True)

#1 forward path
h1 = w1 * i1 + w2 * i2
h2 = w3 * i1 + w4 * i2
pred = w5 * h1 + w6 * h2
print('h1=%f'%(h1))  # change f to d will print an integer
print('h2=%f'%(h2))
print('pred=%f'%(pred))

#2 loss compute
loss = 0.5 * (1- pred)**2
print('loss=%f'%(loss))

#3 backward path
loss.backward()


print('w1 grad=%f'%(w1.grad))
print('w2 grad=%f'%(w2.grad))
print('w3 grad=%f'%(w3.grad))
print('w4 grad=%f'%(w4.grad))
print('w5 grad=%f'%(w5.grad))
print('w6 grad=%f'%(w6.grad))

lr = 0.05

w1 = w1 - lr * (w1.grad.detach())
# By detaching the gradient,
# it's telling PyTorch that the update operation is not
# a part of the forward or backward pass
# and should not be considered during future gradient calculations.
w2 = w2 - lr * (w2.grad.detach())
w3 = w3 - lr * (w3.grad.detach())
w4 = w4 - lr * (w4.grad.detach())
w5 = w5 - lr * (w5.grad.detach())
w6 = w6 - lr * (w6.grad.detach())

print('w1=%f'%(w1))  # change f to d will print an integer
print('w2=%f'%(w2))
print('w3=%f'%(w3))
print('w4=%f'%(w4))
print('w5=%f'%(w5))
print('w6=%f'%(w6))


# iterate 3 times

w1 = torch.tensor([0.11], requires_grad=True)
w2 = torch.tensor([0.21], requires_grad=True)
w3 = torch.tensor([0.12], requires_grad=True)
w4 = torch.tensor([0.08], requires_grad=True)
w5 = torch.tensor([0.14], requires_grad=True)
w6 = torch.tensor([0.15], requires_grad=True)

import torch

lr = 0.05

for _ in range(3):  # iterate 3 times
    i1 = torch.tensor(2.)
    i2 = torch.tensor(3.)

    # Forward path
    h1 = w1 * i1 + w2 * i2
    h2 = w3 * i1 + w4 * i2
    pred = w5 * h1 + w6 * h2

    # Compute loss
    loss = 0.5 * (1 - pred) ** 2

    # Backward path
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        w3 -= lr * w3.grad
        w4 -= lr * w4.grad
        w5 -= lr * w5.grad
        w6 -= lr * w6.grad

    print('w1=%f' % (w1))
    print('w2=%f' % (w2))
    print('w3=%f' % (w3))
    print('w4=%f' % (w4))
    print('w5=%f' % (w5))
    print('w6=%f' % (w6))

h1 = w1 * i1 + w2 * i2
h2 = w3 * i1 + w4 * i2
pred = w5 * h1 + w6 * h2
print('pred 2 = %f' %pred)