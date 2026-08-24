import random
from engine import Value


class Module:
    """Classe de base : gère le zero_grad et l'accès aux paramètres."""

    def zero_grad(self):
        """Met .grad = 0 pour tous les paramètres."""
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """Retourne les paramètres du module."""
        return []


class Neuron(Module):
    def __init__(self, nin, activation="sigmoid"):
        """
        Un neurone :

            z = sum(w_i * x_i) + b

        puis éventuellement une activation.
        """
        self.w = [
            Value(random.uniform(-1.0, 1.0))
            for _ in range(nin)
        ]

        self.b = Value(0.0)

        if activation not in ("sigmoid", "tanh", "linear"):
            raise ValueError(
                f"Unknown activation: {activation}"
            )

        self.activation = activation

    def __call__(self, x):
        """
        Forward pass du neurone.
        """

        # On accepte aussi bien :
        # [Value(...), Value(...)]
        # que :
        # [1.0, 2.0]
        x = [
            xi if isinstance(xi, Value) else Value(xi)
            for xi in x
        ]

        if len(x) != len(self.w):
            raise ValueError(
                f"Expected {len(self.w)} inputs, got {len(x)}"
            )

        # z = sum(w_i * x_i) + b
        act = self.b

        for wi, xi in zip(self.w, x):
            act = act + wi * xi

        # Activation
        if self.activation == "sigmoid":
            return act.sigmoid()

        elif self.activation == "tanh":
            return act.tanh()

        elif self.activation == "linear":
            return act

    def parameters(self):
        """Retourne tous les paramètres du neurone."""
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, activation="sigmoid"):
        self.neurons = [
            Neuron(nin, activation)
            for _ in range(nout)
        ]

    def __call__(self, x):
        """
        Applique tous les neurones à la même entrée.
        """

        outputs = [
            neuron(x)
            for neuron in self.neurons
        ]

        return outputs

    def parameters(self):
        """Concatène les paramètres de tous les neurones."""
        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params


class MLP(Module):
    def __init__(self, nin, nouts, activations=None):
        """
        Exemple :

            MLP(
                nin=2,
                nouts=[4, 4, 1],
                activations=["tanh", "tanh", "linear"]
            )

        donne :

            2 -> 4 -> 4 -> 1
        """

        if len(nouts) == 0:
            raise ValueError("nouts cannot be empty")

        # Activations par défaut
        if activations is None:
            activations = [
                "sigmoid"
                for _ in nouts
            ]

            # Sortie linéaire par défaut
            activations[-1] = "linear"

        if len(activations) != len(nouts):
            raise ValueError(
                "activations and nouts must have the same length"
            )

        self.layers = []

        current_nin = nin

        for nout, activation in zip(nouts, activations):
            layer = Layer(
                current_nin,
                nout,
                activation
            )

            self.layers.append(layer)

            # La sortie de cette couche devient
            # l'entrée de la suivante.
            current_nin = nout

    def __call__(self, x):
        """
        Forward pass dans toutes les couches.
        """

        for layer in self.layers:
            x = layer(x)

        # Pour la sortie finale à un neurone,
        # on retourne directement le Value.
        if len(x) == 1:
            return x[0]

        return x

    def parameters(self):
        """Retourne tous les paramètres du réseau."""
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params
