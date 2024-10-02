class Value:
    """scalar value that stores data, gradient"""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self.prev = set(_children)
        self.op = _op
        self._backward = lambda: None
        self.label = label

    def __repr__(self) -> str:
        return f"Value:(data={self.data}, grad={self.grad}, label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    # returns self+other when given other+self
    def __radd__(self, other):
        return self + other

    # returns self*other when given other*self
    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self - other

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self, other), "**")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other / self

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "RELU")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topological_order = []
        visited = set()

        def build_order_dfs(node):
            if node not in visited:
                topological_order.append(node)
                visited.add(node)
                for child in node.prev:
                    build_order_dfs(child)

        build_order_dfs(self)

        self.grad = 1

        for node in topological_order:
            node._backward()
