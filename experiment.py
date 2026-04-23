"""
CDS521 Dissertation — Model Attack & Defense (MNIST)
Auto-detects CUDA; runs 7 experiments; writes figures/tables to ./outputs/.
CPU expected runtime ~20 min,  GPU expected ~3 min.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------- setup ----------
torch.manual_seed(0); np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> running on {DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

DATA_DIR = "./data"
OUT_DIR  = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH      = 128
EPOCHS     = 3
PGD_ITERS  = 20 if DEVICE == "cuda" else 7
EVAL_N     = 10000 if DEVICE == "cuda" else 2000
EPS_GRID   = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ---------- data ----------
tf = transforms.ToTensor()
train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tf)
test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=256,   shuffle=False)
eval_loader  = DataLoader(Subset(test_set, list(range(EVAL_N))), batch_size=256, shuffle=False)

# ---------- model ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.f1 = nn.Linear(64 * 7 * 7, 128)
        self.f2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))

# ---------- attacks ----------
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    return (x + eps * grad.sign()).clamp(0, 1).detach()

def pgd(model, x, y, eps, alpha=0.01, iters=PGD_ITERS, random_start=False):
    x_orig = x.clone().detach()
    if random_start:
        x_adv = (x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)).clamp(0, 1).detach()
    else:
        x_adv = x_orig.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0, 1).detach()
    return x_adv

# ---------- training ----------
ADV_TRAIN_EPOCHS = 5  # adversarial training needs more epochs to converge

def train(model, loader, epochs=EPOCHS, adv_eps=0.0):
    """Standard training (adv_eps=0) or Madry-style PGD-7 adversarial training."""
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    eps_ = epochs if adv_eps == 0 else ADV_TRAIN_EPOCHS
    for ep in range(eps_):
        model.train(); t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if adv_eps > 0:
                # PGD-7 with random start, step = eps/4 (Madry-style adversarial training)
                x = pgd(model, x, y, adv_eps, alpha=adv_eps / 4, iters=7, random_start=True)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{eps_}  time={time.time()-t0:.1f}s")

def accuracy(model, loader, attack=None, **kw):
    model.eval(); ok = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if attack is not None:
            x = attack(model, x, y, **kw)
        with torch.no_grad():
            ok += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return ok / n

# ---------- E1: train clean ----------
print("\n[E1] Training clean CNN")
net = SimpleCNN()
t = time.time(); train(net, train_loader); e1_time = time.time() - t
clean_acc = accuracy(net, test_loader)
print(f"    clean test acc = {clean_acc:.4f}  |  training time = {e1_time:.1f}s")

# ---------- E2: FGSM sweep on clean ----------
print("\n[E2] FGSM sweep (clean model)")
fgsm_clean = [accuracy(net, eval_loader, attack=(None if e == 0 else fgsm), eps=e) for e in EPS_GRID]
print("    " + ", ".join(f"{e:.2f}:{a:.3f}" for e, a in zip(EPS_GRID, fgsm_clean)))

# ---------- E3: PGD sweep on clean ----------
print("\n[E3] PGD sweep (clean model)")
pgd_clean  = [accuracy(net, eval_loader, attack=(None if e == 0 else pgd), eps=e) for e in EPS_GRID]
print("    " + ", ".join(f"{e:.2f}:{a:.3f}" for e, a in zip(EPS_GRID, pgd_clean)))

# ---------- E4: adversarial training ----------
print("\n[E4] Adversarial training (PGD-7, eps=0.3, Madry-style)")
net_rob = SimpleCNN()
t = time.time(); train(net_rob, train_loader, adv_eps=0.3); e4_time = time.time() - t
fgsm_rob = [accuracy(net_rob, eval_loader, attack=(None if e == 0 else fgsm), eps=e) for e in EPS_GRID]
pgd_rob  = [accuracy(net_rob, eval_loader, attack=(None if e == 0 else pgd), eps=e) for e in EPS_GRID]
rob_clean_acc = accuracy(net_rob, test_loader)
print(f"    robust model clean acc = {rob_clean_acc:.4f}  |  adv training time = {e4_time:.1f}s")
print("    FGSM/rob: " + ", ".join(f"{e:.2f}:{a:.3f}" for e, a in zip(EPS_GRID, fgsm_rob)))
print("    PGD /rob: " + ", ".join(f"{e:.2f}:{a:.3f}" for e, a in zip(EPS_GRID, pgd_rob)))

# ---------- E5 & E6: confusion + per-class @ eps=0.2 ----------
print("\n[E5/E6] Confusion matrix + per-class accuracy (eps=0.2, clean model, FGSM)")
net.eval()
y_true, y_pred_clean, y_pred_adv = [], [], []
for x, y in eval_loader:
    x, y = x.to(DEVICE), y.to(DEVICE)
    x_adv = fgsm(net, x, y, 0.2)
    with torch.no_grad():
        y_pred_clean.extend(net(x).argmax(1).cpu().tolist())
        y_pred_adv.extend(net(x_adv).argmax(1).cpu().tolist())
    y_true.extend(y.cpu().tolist())
cm_clean = confusion_matrix(y_true, y_pred_clean, labels=list(range(10)))
cm_adv   = confusion_matrix(y_true, y_pred_adv,   labels=list(range(10)))
per_cls_clean = cm_clean.diagonal() / cm_clean.sum(1).clip(min=1)
per_cls_adv   = cm_adv.diagonal()   / cm_adv.sum(1).clip(min=1)

# ---------- E7: attack wall-clock cost ----------
print("\n[E7] Attack wall-clock timing (1024 samples)")
x_bench, y_bench = next(iter(DataLoader(Subset(test_set, list(range(1024))), batch_size=1024)))
x_bench, y_bench = x_bench.to(DEVICE), y_bench.to(DEVICE)
# warm-up
_ = fgsm(net, x_bench, y_bench, 0.2); torch.cuda.synchronize() if DEVICE == "cuda" else None
t = time.time(); _ = fgsm(net, x_bench, y_bench, 0.2)
if DEVICE == "cuda": torch.cuda.synchronize()
t_fgsm = time.time() - t
t = time.time(); _ = pgd(net,  x_bench, y_bench, 0.2, iters=7)
if DEVICE == "cuda": torch.cuda.synchronize()
t_pgd7 = time.time() - t
t = time.time(); _ = pgd(net,  x_bench, y_bench, 0.2, iters=20)
if DEVICE == "cuda": torch.cuda.synchronize()
t_pgd20 = time.time() - t
print(f"    FGSM: {t_fgsm*1000:.1f} ms,  PGD-7: {t_pgd7*1000:.1f} ms,  PGD-20: {t_pgd20*1000:.1f} ms")

# ---------- save numerical results ----------
results = {
    "device": DEVICE,
    "eps_grid": EPS_GRID,
    "clean_acc": clean_acc,
    "rob_clean_acc": rob_clean_acc,
    "fgsm_clean": fgsm_clean, "pgd_clean": pgd_clean,
    "fgsm_rob":   fgsm_rob,   "pgd_rob":   pgd_rob,
    "per_class_clean": per_cls_clean.tolist(),
    "per_class_adv_fgsm_eps02": per_cls_adv.tolist(),
    "train_time_clean_s": e1_time,
    "train_time_adv_s":   e4_time,
    "attack_time_1024_s": {"fgsm": t_fgsm, "pgd7": t_pgd7, "pgd20": t_pgd20},
    "pgd_iters_default": PGD_ITERS,
    "eval_n": EVAL_N,
}
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------- Figure 1: adversarial samples ----------
x_demo, y_demo = next(iter(eval_loader))
x_demo, y_demo = x_demo.to(DEVICE), y_demo.to(DEVICE)
x_adv_demo = fgsm(net, x_demo, y_demo, 0.25)
with torch.no_grad():
    p_c = net(x_demo).argmax(1); p_a = net(x_adv_demo).argmax(1)
idxs = [i for i in range(len(y_demo)) if p_c[i] == y_demo[i] and p_a[i] != y_demo[i]][:2]

fig, ax = plt.subplots(2, 3, figsize=(7, 4.5))
for row, i in enumerate(idxs):
    orig = x_demo[i, 0].cpu().numpy()
    adv  = x_adv_demo[i, 0].cpu().numpy()
    ax[row, 0].imshow(orig, cmap="gray"); ax[row, 0].set_title(f"clean  -> pred {p_c[i].item()}")
    ax[row, 1].imshow(adv - orig, cmap="seismic", vmin=-0.3, vmax=0.3)
    ax[row, 1].set_title(r"perturbation  ($\varepsilon$=0.25)")
    ax[row, 2].imshow(adv, cmap="gray"); ax[row, 2].set_title(f"adversarial  -> pred {p_a[i].item()}")
    for j in range(3): ax[row, j].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig1_samples.png", dpi=150); plt.close()

# ---------- Figure 2: two-panel (curves + per-class) ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(EPS_GRID, fgsm_clean, "o-",  label="FGSM / standard")
ax[0].plot(EPS_GRID, pgd_clean,  "s-",  label=f"PGD-{PGD_ITERS} / standard")
ax[0].plot(EPS_GRID, fgsm_rob,   "o--", label="FGSM / adv-trained")
ax[0].plot(EPS_GRID, pgd_rob,    "s--", label=f"PGD-{PGD_ITERS} / adv-trained")
ax[0].set_xlabel(r"perturbation budget  $\varepsilon$  ($L_\infty$)")
ax[0].set_ylabel("test accuracy"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
ax[0].set_title("(a) robustness curves")

cls = np.arange(10); w = 0.4
ax[1].bar(cls - w/2, per_cls_clean, w, label="clean")
ax[1].bar(cls + w/2, per_cls_adv,   w, label=r"FGSM, $\varepsilon$=0.2")
ax[1].set_xticks(cls); ax[1].set_xlabel("digit class"); ax[1].set_ylabel("accuracy")
ax[1].legend(fontsize=8); ax[1].set_title("(b) per-class accuracy")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig2_curves.png", dpi=150); plt.close()

# ---------- Figure 3: confusion matrices ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
sns.heatmap(cm_clean, annot=False, cmap="Blues",  ax=ax[0], cbar=False)
ax[0].set_title("(a) clean predictions");       ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
sns.heatmap(cm_adv,   annot=False, cmap="Reds",  ax=ax[1], cbar=False)
ax[1].set_title(r"(b) FGSM, $\varepsilon$=0.2"); ax[1].set_xlabel("predicted"); ax[1].set_ylabel("true")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig3_confusion.png", dpi=150); plt.close()

# ---------- Table of compute cost ----------
with open(f"{OUT_DIR}/table_cost.csv", "w") as f:
    f.write("attack,iterations,wall_clock_s_per_1024,relative_cost\n")
    f.write(f"FGSM,1,{t_fgsm:.4f},1.00\n")
    f.write(f"PGD,7,{t_pgd7:.4f},{t_pgd7/t_fgsm:.2f}\n")
    f.write(f"PGD,20,{t_pgd20:.4f},{t_pgd20/t_fgsm:.2f}\n")

print("\n>>> all done. outputs written to ./outputs/")
print(f"    clean acc={clean_acc:.4f}  robust clean acc={rob_clean_acc:.4f}")
print(f"    FGSM@eps=.3 std/rob = {fgsm_clean[-1]:.3f} / {fgsm_rob[-1]:.3f}")
print(f"    PGD @eps=.3 std/rob = {pgd_clean[-1]:.3f} / {pgd_rob[-1]:.3f}")
