import learntorch
import csv
import time

# 1. Función de Carga (Data Loader)
def load_mnist(filename, limit=None):
    images, labels = [], []
    print(f"Cargando {filename}...")
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader) # header
        for i, row in enumerate(reader):
            if limit and i >= limit: break
            
            # Label One-Hot: 3 -> [0,0,0,1,0...]
            label = int(row[0])
            target = [0.0] * 10
            target[label] = 1.0
            
            # Pixels Normalize: 0-255 -> 0.0-1.0
            img = [float(x)/255.0 for x in row[1:]]
            
            images.append(img)
            labels.append(target)
    return images, labels

# --- PREPARACIÓN DE DATOS ---
print("--- 1. LOADING DATA ---")
# Cargamos datos (por ejemplo 1000 muestras para probar rápido)
LIMIT = 1000
raw_images, raw_labels = load_mnist('test/mnist_train.csv', limit=LIMIT)

# Convertimos a una lista plana de floats para crear el Tensor
# X: (N, 784)
flat_images = [pixel for img in raw_images for pixel in img]
x_train = learntorch.Tensor([len(raw_images), 784], flat_images)

# Y: (N, 10)
flat_labels = [val for label in raw_labels for val in label]
y_train = learntorch.Tensor([len(raw_labels), 10], flat_labels)

print(f"Datos cargados. X shape: {x_train.shape}, Y shape: {y_train.shape}")

# --- DEFINICIÓN DEL MODELO ---
print("\n--- 2. CREATING MODEL ---")
# Arquitectura: Linear(784->128) -> ReLU -> Linear(128->10)
model = learntorch.Serial([
    learntorch.Linear(784, 128),
    learntorch.ReLU(),
    learntorch.Linear(128, 10),
    learntorch.Softmax()
])

# --- CONFIGURACIÓN ---
print("\n--- 3. CONFIGURATION ---")
# Optimizador SGD con Learning Rate 0.1
optimizer = learntorch.SGD(model.parameters(), lr=0.1)

# Función de pérdida: CrossEntropy (ideal para clasificación) o MSELoss
criterion = learntorch.CrossEntropy()
# criterion = learntorch.MSELoss() 

# --- TRAINING ---
print("\n--- 4. TRAINING (Stochastic Gradient Descent - Batch Size = 1) ---")
# trainer = learntorch.Trainer(model, optimizer, criterion)
# En C++ el Trainer hace full-batch, así que hacemos un bucle manual en Python
# para simular Batch Size = 1

start_time = time.time()

num_samples = len(raw_images)
epochs = 5 # Reducimos epochs porque SGD es más lento por iteración
print(f"Training on {num_samples} samples for {epochs} epochs...")

for epoch in range(epochs):
    epoch_loss = 0.0
    
    for i in range(num_samples):
        # 1. Prepare single-sample batch
        # Slice raw data for this sample
        start_idx = i * 784
        end_idx = start_idx + 784
        x_sample_data = flat_images[start_idx:end_idx]
        
        start_idy = i * 10
        end_idy = start_idy + 10
        y_sample_data = flat_labels[start_idy:end_idy]
        
        # Create Tensors (1, 784) and (1, 10)
        x_batch = learntorch.Tensor([1, 784], x_sample_data)
        y_batch = learntorch.Tensor([1, 10], y_sample_data)
        
        # 2. Zero Grad
        optimizer.zero_grad()
        
        # 3. Forward
        # Model expects vector<Tensor>, and returns vector<Tensor>
        outputs = model.forward([x_batch])
        prediction = outputs[0]
        
        # 4. Loss
        loss = criterion.forward(prediction, y_batch)
        
        # Accumulate for printing
        # loss.item() reads the scalar value
        epoch_loss += loss.item()
        
        # 5. Backward
        loss.backward()
        
        # 6. Step
        optimizer.step()
        
    avg_loss = epoch_loss / num_samples
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

end_time = time.time()

print(f"\nTiempo total de entrenamiento: {end_time - start_time:.2f} segundos")
print("¡Entrenamiento completado!")

# --- VALIDACIÓN (Predicción rápida) ---
print("\n--- 5. CHECK PREDICTION (First Sample) ---")
# Hacemos forward manually para ver la predicción del primer ejemplo
# Ojo: Trainer fit ya hizo forward, pero hacemos uno limpio
pred = model.forward([x_train])[0] # Model devuelve lista de tensores

# Vemos la predicción para la primera imagen
print("Target real (One-Hot):")
first_target = raw_labels[0]
print(first_target)
print(f"Clase real: {first_target.index(1.0)}")

print("Logits Predichos (Primeros 10 valores correspondientes a 0-9):")
# Extraemos los datos del tensor devuelto
pred_data = pred.data()
first_pred_logits = pred_data[0:10]
print([f"{x:.4f}" for x in first_pred_logits])

# Argmax simple en Python
max_val = max(first_pred_logits)
pred_class = first_pred_logits.index(max_val)
print(f"Clase Predicha: {pred_class}")
