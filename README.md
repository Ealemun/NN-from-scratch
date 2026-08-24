# 🧠 Mini Autograd & MLP from Scratch

Implémentation pédagogique d'un **moteur d'autograd** et d'un **MLP (Multi-Layer Perceptron)** en Python, sans utiliser PyTorch pour les calculs du modèle.

L'objectif est de comprendre concrètement les mécanismes derrière la **backpropagation**, les **computational graphs**, les **gradients** et la **descente de gradient**.

## 📁 Structure

```text
.
├── engine.py          # Moteur d'autograd
├── nn.py              # Neuron, Layer et MLP
├── train.py           # Entraînement sur XOR
├── test_engine.py     # Tests et comparaison avec PyTorch
└── README.md
```

## ⚙️ Autograd

`engine.py` contient une classe `Value` représentant un scalaire dans un computational graph.

Chaque valeur stocke notamment :

- sa valeur numérique (`data`)
- son gradient (`grad`)
- ses parents dans le graphe
- l'opération qui l'a créée
- sa fonction locale de backpropagation

Les opérations supportées incluent :

```text
+   -   *   /   **
sigmoid
tanh
```

Le moteur utilise la **reverse-mode automatic differentiation**.

Par exemple :

```python
x = Value(3.0)

y = (x * 2 + 1) ** 2

y.backward()

print(x.grad)
```

L'appel à `backward()` construit un ordre topologique du graphe puis propage les gradients de la sortie vers les entrées en appliquant la règle de chaîne.

## 🧠 Neural Network

`nn.py` construit les composants du réseau à partir de `Value`.

### Neuron

Un neurone calcule :

\[
z = \sum_i w_i x_i + b
\]

puis applique une fonction d'activation :

- `sigmoid`
- `tanh`
- `linear`

### Layer

Une couche contient plusieurs neurones appliqués à la même entrée.

### MLP

Le `MLP` permet d'empiler plusieurs couches.

Exemple :

```python
model = MLP(
    nin=2,
    nouts=[4, 1],
    activations=["tanh", "sigmoid"]
)
```

Architecture :

```text
2 inputs
   ↓
4 neurons + tanh
   ↓
1 neuron + sigmoid
   ↓
output
```

## 🎯 Entraînement

Le modèle est entraîné sur le problème classique **XOR** :

```python
xs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

ys = [0, 1, 1, 0]
```

La boucle d'entraînement suit le processus classique :

```text
Forward
   ↓
Prediction
   ↓
Loss
   ↓
Backward
   ↓
Gradients
   ↓
Parameter update
   ↓
Repeat
```

La loss utilisée est la somme des erreurs quadratiques :

\[
L = \sum_i (\hat{y}_i-y_i)^2
\]

Les paramètres sont ensuite mis à jour avec une simple **gradient descent (SGD)** :

```python
for p in model.parameters():
    p.data -= lr * p.grad
```

## 🧪 Tests

Les tests utilisent PyTorch comme **référence indépendante** pour vérifier l'implémentation de l'autograd.

Ils permettent notamment de vérifier :

- les opérations élémentaires (`+`, `*`, `**`)
- la règle de chaîne
- `sigmoid`
- les gradients
- le comportement du MLP

L'objectif est de vérifier que les gradients produits par notre moteur correspondent à ceux de PyTorch.

Lancer les tests :

```bash
pytest -v
```

## 🚀 Installation

Cloner le repository puis installer les dépendances :

```bash
pip install torch pytest
```

Lancer l'entraînement :

```bash
python train.py
```

Lancer les tests :

```bash
pytest -v
```

## 📚 Concepts étudiés

Ce projet permet de mettre en pratique :

- Computational graphs
- Automatic differentiation
- Reverse-mode autodiff
- Chain rule
- Backpropagation
- Gradients
- Activation functions
- Neural networks
- Gradient descent
- Validation numérique avec PyTorch

## 🔬 Limites

Le moteur est volontairement minimaliste et travaille avec des **valeurs scalaires**.

Contrairement à PyTorch, il ne gère pas encore les tensors, les opérations vectorisées ou les architectures complexes.

Le but n'est donc pas de remplacer un framework de deep learning, mais de comprendre les mécanismes fondamentaux qui se cachent derrière :

```python
loss.backward()
```

## 🗺️ Extensions possibles

Quelques pistes pour aller plus loin :

- Implémenter Momentum
- Implémenter RMSProp / Adam
- Ajouter le support des vecteurs et matrices
- Implémenter une classe `Tensor`
- Ajouter les mini-batches
- Ajouter d'autres fonctions d'activation
- Automatiser la comparaison des gradients avec PyTorch

---

**Projet pédagogique — implémentation d'un mini moteur d'autograd et d'un MLP from scratch en Python.**
