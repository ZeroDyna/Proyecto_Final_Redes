import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import ucsp

### LANZEN LOS 4 WORKERS
WORKERS = [
    ("127.0.0.1", 9001),
    ("127.0.0.1", 9002),
    ("127.0.0.1", 9003),
    ("127.0.0.1", 9004)
]

#  ./worker <puerto> <SAMPLES_PER_WORKER>
S = 16
N = len(WORKERS)

class MulticlassClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden1: int = 128, hidden2: int = 64, hidden3: int = 32):
        super(MulticlassClassifier, self).__init__()
        self.f1 = nn.Linear(input_dim, hidden1)
        self.f2 = nn.Linear(hidden1, hidden2)
        self.f3 = nn.Linear(hidden2, hidden3)
        self.logits = nn.Linear(hidden3, num_classes)
        self.logvars = nn.Linear(hidden3, num_classes)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        logits2 = self.logits(x)
        logvars2 = self.logvars(x)
        return logits2, logvars2

### No tocar xd
def ordered_params(model):
    return [
        model.f1.weight,
        model.f1.bias,
        model.f2.weight,
        model.f2.bias,
        model.f3.weight,
        model.f3.bias,
        model.logits.weight,
        model.logits.bias,
        model.logvars.weight,
        model.logvars.bias,
    ]


def extract_weights(model):
    parts = []
    for p in ordered_params(model):
        parts.append(p.detach().cpu().numpy().astype(np.float32).ravel())
    return np.concatenate(parts)


def load_gradients(model, grad_avg):
    grad = np.asarray(grad_avg, dtype=np.float32)
    offset = 0
    for p in ordered_params(model):
        n = p.numel()
        chunk = grad[offset:offset + n].reshape(p.shape)
        p.grad = torch.from_numpy(chunk.copy())
        offset = offset + n


def build_worker_buffers(model_buffer, batch_x, batch_y):
    x = batch_x.cpu().numpy().astype(np.float32)
    y = batch_y.cpu().numpy().astype(np.float32)
    buffers = []
    for w in range(N):
        start = w * S
        end = start + S
        samples = np.concatenate([x[start:end], y[start:end]], axis=1).astype(np.float32).ravel()
        buf = np.concatenate([model_buffer, samples]).astype(np.float32)
        buffers.append(buf)
    return buffers


torch.manual_seed(42)
nsamples = 1000
diminput = 14
nclasses = 3
batchsize = (N + 1) * S   # Master(S) + worker1(S) + worker2(S) + worker3(S)

### DATASET
CSV = "Dataset of Diabetes.csv"

df = pd.read_csv(CSV, header=None, skiprows=1)

xnp = df.iloc[:, :diminput].values.astype(np.float32)
ynp = df.iloc[:, -nclasses:].values.astype(np.float32)

x = torch.tensor(xnp)
y = torch.tensor(ynp)
print(f"y: {y.size()} x: {x.size()}")

dataset = TensorDataset(x, y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

traindataset, testdataset = random_split(dataset, [train_size, test_size])
loader = DataLoader(traindataset, batch_size=batchsize, shuffle=True, drop_last=True)
testloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False)

model = MulticlassClassifier(input_dim=diminput, num_classes=nclasses)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

### CANTIDAD DE EPOCAS
epochs = 360
###

true = []
pred = []
train_tracker = []
test_tracker = []
accuracy_tracker = []

### Para que el master trabaje xd
def master_gradient(model):
    parts = []
    for p in ordered_params(model):
        if p.grad is not None:
            parts.append(p.grad.cpu().numpy().astype(np.float32).ravel())
        else:
            parts.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(parts)

def master_step(model, optimizer, criterion, batchx, batchy):
    mx = batchx[:S]
    my = batchy[:S]
    lmout, _ = model(mx)
    lm = criterion(lmout, my)
    lm.backward()

    mgrad = master_gradient(model)
    #print(f"[MASTER] Gradient calculated loss: {lm.item():.4f}")
    optimizer.zero_grad()

    return mgrad

def workers_step(model, batchx, batchy):
    wx = batchx[S:]
    wy = batchy[S:]
    Buffer = extract_weights(model)
    Buffers = build_worker_buffers(Buffer, wx, wy)

    ### PESOS ENVIADOS
    print(f"[MASTER] Primeros 5 pesos enviados al worker 0: {Buffers[0][:5]}")

    wgrad = np.asarray(
        ucsp.distribute(Buffers, WORKERS, ucsp.TOTAL_MODEL_PARAMS),
        dtype=np.float32
    )

    print(f"[MASTER] Primeros 5 grads recibidos: {wgrad[:5]}")
    print()
    print()
    print()
    #print(f"[MASTER] Suma total del gradiente: {np.sum(np.abs(wgrad)):.8f}")

    return wgrad

def train_batch(model, optimizer, criterion, batchx, batchy):
    optimizer.zero_grad()
    Mgrad = master_step(model, optimizer, criterion, batchx, batchy)
    Wgrad = workers_step(model, batchx, batchy)
    GAVG = (Mgrad + Wgrad) / 2.0
    load_gradients(model, GAVG)
    optimizer.step()

    with torch.no_grad():
        logits, lv = model(batchx)
        loss = criterion(logits, batchy)

    return loss.item()

def evaluate(model, criterion, testloader, epoch, epochs, true, pred):
    model.eval()
    vloss = 0
    total = 0
    ncorrect = 0

    with torch.no_grad():
        for batchx, batchy in testloader:
            logits, lv = model(batchx)
            loss = criterion(logits, batchy)
            vloss = vloss + loss.item()
            predictions = torch.argmax(logits, dim=1)
            total = total + batchx.size(0)
            ncorrect = ncorrect + (predictions == torch.argmax(batchy, dim=1)).sum().item()
            
            if epoch == epochs - 1:
                true.extend(torch.argmax(batchy, dim=1).tolist())
                pred.extend(predictions.tolist())
                
    return vloss / len(testloader), ncorrect / total

### EPOCAS

for i in range(epochs):
    model.train()
    eloss = 0
 
    for batchx, batchy in loader:
        eloss = eloss + train_batch(model, optimizer, criterion, batchx, batchy)
 
    train_tracker.append(eloss / len(loader))
    print(f"Epoch: {i + 1}/{epochs} | Loss: {train_tracker[-1]:.6f} | ", end="")
 
    test_loss, accuracy = evaluate(model, criterion, testloader, i, epochs, true, pred)
    test_tracker.append(test_loss)
    accuracy_tracker.append(accuracy)
    print(f"Test loss: {test_loss:.4f} | Accuracy : {accuracy:.4f}")


### GRAFICO

plt.figure(figsize=(8, 4))
plt.plot(train_tracker, marker="o")
plt.title("Loss X Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("Gloss.png")
plt.close()

plt.figure()
plt.plot(train_tracker, label="Training loss")
plt.plot(test_tracker, label="Test loss")
plt.plot(accuracy_tracker, label="Precision")
plt.legend()
plt.savefig("Comparison.png")
plt.close()


### MATRIZ DE CONFUSION

MC = confusion_matrix(true, pred)
DISP = ConfusionMatrixDisplay(confusion_matrix=MC)
DISP.plot(cmap=plt.cm.Blues)

plt.title("CM")
plt.tight_layout()

plt.savefig("CM.png")
plt.close()

print("Reporte:")
print(classification_report(true, pred, digits=3))