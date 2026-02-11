import learntorch
import time

print("=== Custom Framework Verification (Batch=4, Steps=3) ===\n")

# 1. Datos (Batch Size = 4, Features = 2)
# X shape (4, 2)
X_data = [
    1.0, 2.0,
    0.5, 0.5,
    -1.0, 0.0,
    0.0, 1.0
]
X = learntorch.Tensor([4, 2], X_data)

# Targets One-Hot (4, 2)
# Indices: 0, 1, 0, 1
y_data = [
    1.0, 0.0,  # Class 0
    0.0, 1.0,  # Class 1
    1.0, 0.0,  # Class 0
    0.0, 1.0   # Class 1
]
y = learntorch.Tensor([4, 2], y_data)

# 2. Modelo
linear = learntorch.Linear(2, 2)

# Inicialización Manual Determinista
# Nota: Asumimos que podemos acceder a los parámetros así. 
# Si Linear expone W y B como propiedades en los bindings.
# Usamos linear.parameters() que devuelve una lista de tensores [W, B] usualmente.
params = linear.parameters()
W = params[0] # (Out, In) -> (2, 2)
b = params[1] # (1, Out) -> (1, 2)

# Setear datos manualmente accediendo al array interno (si es posible desde Python)
# Ojo: learntorch.Tensor probablemente no expone un setter directo elemento a elemento eficiente.
# Asumiremos que podemos pasarle una lista nueva a un hipotético método setData o re-crearlo.
# Si no, tendremos que confiar en que la inicialización aleatoria difiera o 
# intentar hackearlo.
# Pero espera! En test_learntorch.py no se muestra cómo setear pesos manualmente.
# Como es bindings con pybind11, usualmente podemos asignar si está expuesto el buffer protocol o similar.

# Intento de re-asignar los tensores internos si el binding expone los atributos W y B
# Si linear.W es read-write property:

# W = [[0.1, 0.2], [0.3, 0.4]]
# Flattened: [0.1, 0.2, 0.3, 0.4]
new_W = learntorch.Tensor([2, 2], [0.1, 0.2, 0.3, 0.4])
# b = [0.1, -0.1] -> Broadcasted to [1, 2] usually or just [2]
new_b = learntorch.Tensor([1, 2], [0.1, -0.1])

# Hay que ver si Linear tiene métodos para reemplazar sus pesos o si accedemos al atributo
# Vamos a asumir que podemos escribir en `linear.W` y `linear.B` si están expuestos. 
# Si no, este script fallará y lo arreglaremos viendo los bindings.
linear.W = new_W
linear.B = new_b

print("Initial W (via get_data if available):")
print(linear.W.data()) 
print("Initial b:")
print(linear.B.data())
print("-" * 50)

# 3. Optimizador
optimizer = learntorch.SGD(linear.parameters(), 0.1)

# Layers
softmax = learntorch.Softmax()
criterion = learntorch.CrossEntropy()

# 4. Training Loop
for step in range(1, 4):
    print(f"\nStep {step}:")
    
    optimizer.zero_grad()
    
    # Forward
    # linear.forward espera una lista de tensores [input]
    out_linear = linear.forward([X])[0]
    
    # Softmax
    probs = softmax.forward([out_linear])[0]
    
    # Loss
    loss_val = criterion.forward(probs, y)
    
    # Backward
    # loss_val es un Tensor escalar (1,1) o similar
    loss_val.backward()
    
    # Imprimir Loss
    print(f"  Loss: {loss_val.data()[0]:.6f}")
    
    # Gradients
    print("\n  Gradients:")
    # linear.W.grad es un Tensor
    if linear.W.grad:
        print(f"  Grad W: {linear.W.grad.data()}")
    if linear.B.grad:
        print(f"  Grad b: {linear.B.grad.data()}")
        
    # Update
    optimizer.step()
    
    print("\n  Weights After Update:")
    print(f"  W: {linear.W.data()}")
    print(f"  b: {linear.B.data()}")
    print("-" * 50)
