import torch

from engine import Value

def test_add():
    x_mine = Value(2.0)
    y_mine = Value(3.0)

    out_mine = x_mine + y_mine
    out_mine.backward()

    assert abs(out_mine.data - 5.0) < 1e-6
    assert abs(x_mine.grad - 1.0) < 1e-6
    assert abs(y_mine.grad - 1.0) < 1e-6


def test_mul():
    x_mine = Value(2.0)
    y_mine = Value(3.0)

    out_mine = x_mine * y_mine
    out_mine.backward()

    assert abs(out_mine.data - 6.0) < 1e-6
    assert abs(x_mine.grad - 3.0) < 1e-6
    assert abs(y_mine.grad - 2.0) < 1e-6


def test_pow():
    x_mine = Value(3.0)

    out_mine = x_mine ** 2
    out_mine.backward()

    assert abs(out_mine.data - 9.0) < 1e-6
    assert abs(x_mine.grad - 6.0) < 1e-6


def test_chain_rule():
    # y = (x * 2 + 1)^2
    #
    # dy/dx = 2 * (2x + 1) * 2
    #
    # à x = 3 :
    # y = 49
    # dy/dx = 28

    x = Value(3.0)

    y = (x * 2 + 1) ** 2
    y.backward()

    assert abs(y.data - 49.0) < 1e-6
    assert abs(x.grad - 28.0) < 1e-6


def test_sigmoid_matches_pytorch():
    x_mine = Value(0.7)

    x_torch = torch.tensor(
        0.7,
        dtype=torch.float64,
        requires_grad=True
    )

    y_mine = x_mine.sigmoid()
    y_torch = torch.sigmoid(x_torch)

    y_mine.backward()
    y_torch.backward()

    assert abs(y_mine.data - y_torch.item()) < 1e-6
    assert abs(x_mine.grad - x_torch.grad.item()) < 1e-6




def test_sigmoid_matches_pytorch():
    x_mine = Value(0.7)
    x_torch = torch.tensor(0.7, requires_grad=True)

    y_mine = x_mine.sigmoid()
    y_torch = torch.sigmoid(x_torch)

    y_mine.backward()
    y_torch.backward()

    assert abs(y_mine.data - y_torch.item()) < 1e-6
    assert abs(x_mine.grad - x_torch.grad.item()) < 1e-6


def test_mlp_matches_pytorch():
    """
    Compare notre petit MLP avec exactement le même MLP
    construit en PyTorch.

    Architecture :

        2 -> 2 -> 1

    Activations :

        tanh -> sigmoid

    Loss :

        (prediction - target)^2
    """

    # =====================================================
    # Données
    # =====================================================

    x = [0.4, -0.7]
    target = 1.0

    # =====================================================
    # Notre moteur
    # =====================================================

    x1 = Value(x[0])
    x2 = Value(x[1])

    # Première couche
    w11 = Value(0.5)
    w12 = Value(-0.3)
    b1 = Value(0.1)

    w21 = Value(0.8)
    w22 = Value(0.2)
    b2 = Value(-0.2)

    # Deuxième couche
    w31 = Value(0.7)
    w32 = Value(-0.5)
    b3 = Value(0.05)

    # Forward couche 1
    z1 = w11 * x1 + w12 * x2 + b1
    h1 = z1.tanh()

    z2 = w21 * x1 + w22 * x2 + b2
    h2 = z2.tanh()

    # Forward couche 2
    z3 = w31 * h1 + w32 * h2 + b3
    y_mine = z3.sigmoid()

    # Loss
    loss_mine = (y_mine - target) ** 2

    # Backward
    loss_mine.backward()

    # =====================================================
    # PyTorch
    # =====================================================

    tx1 = torch.tensor(x[0], dtype=torch.float64, requires_grad=True)
    tx2 = torch.tensor(x[1], dtype=torch.float64, requires_grad=True)

    tw11 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    tw12 = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
    tb1 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    tw21 = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    tw22 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    tb2 = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)

    tw31 = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    tw32 = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
    tb3 = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

    # Forward couche 1
    tz1 = tw11 * tx1 + tw12 * tx2 + tb1
    th1 = torch.tanh(tz1)

    tz2 = tw21 * tx1 + tw22 * tx2 + tb2
    th2 = torch.tanh(tz2)

    # Forward couche 2
    tz3 = tw31 * th1 + tw32 * th2 + tb3
    y_torch = torch.sigmoid(tz3)

    # Loss
    loss_torch = (y_torch - target) ** 2

    # Backward
    loss_torch.backward()

    # =====================================================
    # Comparaison forward
    # =====================================================

    assert abs(
        y_mine.data - y_torch.item()
    ) < 1e-6

    assert abs(
        loss_mine.data - loss_torch.item()
    ) < 1e-6

    # =====================================================
    # Comparaison gradients
    # =====================================================

    assert abs(w11.grad - tw11.grad.item()) < 1e-6
    assert abs(w12.grad - tw12.grad.item()) < 1e-6
    assert abs(b1.grad - tb1.grad.item()) < 1e-6

    assert abs(w21.grad - tw21.grad.item()) < 1e-6
    assert abs(w22.grad - tw22.grad.item()) < 1e-6
    assert abs(b2.grad - tb2.grad.item()) < 1e-6

    assert abs(w31.grad - tw31.grad.item()) < 1e-6
    assert abs(w32.grad - tw32.grad.item()) < 1e-6
    assert abs(b3.grad - tb3.grad.item()) < 1e-6
