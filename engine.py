import math


class Value:
    """
    A scalar value used to build a computational graph.

    Each Value stores:
    - data: the numerical value
    - grad: the gradient accumulated during backward()
    - _prev: the parent nodes in the computational graph
    - _op: the operation that created this node
    - _backward: local function used to propagate gradients
    """
        
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        out = self + other

        d(out)/d(self) = 1
        d(out)/d(other) = 1
        """
        # Cast automatique : Value(3) + 2 -> Value(3) + Value(2)
        other = other if isinstance(other, Value) else Value(other)

        # Forward
        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        # Backward
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        out = self * other

        d(out)/d(self) = other
        d(out)/d(other) = self
        """
        other = other if isinstance(other, Value) else Value(other)

        # Forward
        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        # Backward
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, exponent):
        """
        out = self ** exponent

        d(x^n)/dx = n * x^(n-1)
        """
        assert isinstance(exponent, (int, float)), \
            "exponent must be an int or float"

        # Forward
        out = Value(
            self.data ** exponent,
            (self,),
            f"**{exponent}"
        )

        # Backward
        def _backward():
            self.grad += (
                exponent
                * self.data ** (exponent - 1)
                * out.grad
            )

        out._backward = _backward

        return out

    def sigmoid(self):
        """
        sigmoid(x) = 1 / (1 + exp(-x))

        sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        """
        # Forward
        s = 1.0 / (1.0 + math.exp(-self.data))

        out = Value(
            s,
            (self,),
            "sigmoid"
        )

        # Backward
        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        """
        tanh'(x) = 1 - tanh(x)^2
        """
        # Forward
        t = math.tanh(self.data)

        out = Value(
            t,
            (self,),
            "tanh"
        )

        # Backward
        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward

        return out

    # ---------------------------------------------------------
    # Opérations dérivées des opérations précédentes
    # ---------------------------------------------------------

    def __neg__(self):
        # -x = x * (-1)
        return self * -1

    def __radd__(self, other):
        # 2 + x -> x + 2
        return self + other

    def __sub__(self, other):
        # x - y = x + (-y)
        return self + (-other)

    def __rsub__(self, other):
        # 2 - x = 2 + (-x)
        return other + (-self)

    def __rmul__(self, other):
        # 2 * x -> x * 2
        return self * other

    def __truediv__(self, other):
        # x / y = x * y^(-1)
        return self * other ** -1

    def backward(self):
        """
        Reverse-mode automatic differentiation.

        1. Construire le graphe en ordre topologique.
        2. Initialiser le gradient de la sortie à 1.
        3. Parcourir le graphe en sens inverse.
        """

        # -----------------------------------------------------
        # 1. Topological sort
        # -----------------------------------------------------

        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topo(child)

                topo.append(node)

        build_topo(self)

        # -----------------------------------------------------
        # 2. dL/dL = 1
        # -----------------------------------------------------

        self.grad = 1.0

        # -----------------------------------------------------
        # 3. Backward pass
        # -----------------------------------------------------

        for node in reversed(topo):
            node._backward()
