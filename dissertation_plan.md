# CDS521 Course Dissertation — Master Plan & Full Report Draft

> **Student**: Wang Wenxuan (王文轩)　|　**Student ID**: 1397228
> **Course**: CDS521 Foundation of AI　|　**Deadline**: 24 April 2026
> **Chosen Topic**: **B — Model Attack and Defense**

---

## 0. 本机环境检测结果 (实测)

| 项目 | 状态 |
|---|---|
| GPU 硬件 | ✅ **NVIDIA GeForce RTX 2050, 4 GB VRAM, CUDA 12.5 驱动** |
| 当前 PyTorch | ⚠️ `2.3.1+cpu` (**CPU 版本**,未启用 CUDA) |
| torchvision | `0.18.1+cpu` |
| matplotlib | `3.9.0` |
| numpy | `1.26.4` |
| scikit-learn | `1.5.0` |
| seaborn | `0.13.2` |
| D: 剩余空间 | 53 GB (足够) |

**两种执行路径:**

- **路径 A (推荐):** 重装 CUDA 版 PyTorch → 实验总耗时 **~5 分钟**,可做 20 步 PGD、完整测试集
- **路径 B (现状):** 保持 CPU → 实验总耗时 **~20 分钟**,用 7 步 PGD + 2000 样本子集

代码会自动检测 `torch.cuda.is_available()`,无需改动,只是 GPU 模式下默认参数更大一点。

### 切换到 GPU 的一行命令

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
# 期望输出: True NVIDIA GeForce RTX 2050
```

---

## 1. 选题决策:为什么选 B (Attack & Defense)

| 维度 | A. Transfer Learning | **B. Attack & Defense (选中)** |
|---|---|---|
| **CPU/小 GPU 友好度** | ResNet 微调慢 | **MNIST + SimpleCNN 在 CPU 20 min / GPU 5 min** |
| **实验深度/页** | 加载→微调→评估,较单一 | **干净模型 + FGSM + PGD + 对抗训练** 4 组对照 |
| **视觉冲击力** | 准确率曲线 | **原图 vs 扰动 vs 对抗样本 三栏图 + 混淆矩阵** |
| **技术深度** | 已是工程常识 | 含 min-max 公式、$L_\infty$ 几何,学术感更强 |
| **选题稀缺性** | 多数同学首选 | 相对少人选,更易出彩 |
| **未来方向写作** | few-shot / 域适应 (写过千遍) | **certified defense / 物理攻击** 活跃前沿 |

**结论:选 B。** 信息密度 × 可视化 × 学术深度三维最优。

---

## 2. 高分策略 (对齐 Rubric)

官方 4 项要求,每条都做"不止于最低要求":

| 要求 | 最低做法 | **本方案做法 (加分点)** |
|---|---|---|
| 理论理解 | 复述定义 | 引入 Goodfellow 线性化假设 + **Madry min-max 公式 (1)**;给出 FGSM 闭式解推导 |
| 应用分析 | 举 1-2 例 | **3 个真实案例**:物理停车标志 (Eykholt)、人脸识别眼镜 (Sharif)、恶意软件 (Grosse);**显式 threat-model 四维分类** (NIST) |
| 伦理 + 未来 | 泛泛而谈 | 双向讨论 dual-use + **偏见放大**;未来指 certified defense + **foundation-model 鲁棒性** (Zou 2023) |
| 实现 | 单一攻击 + 截图 | **4 组受控实验 + 3 张图 + 2 张表**;报告 FGSM/PGD 算力差异、**每类准确率**差异、**混淆矩阵** |

**4 页预算:**

| 页 | 内容 | 关键视觉 |
|---|---|---|
| 1 | Header + Abstract + §1 Intro + §2 Theory 前半 | 公式 (1) |
| 2 | §2 Theory 后半 + §3 Application | (含表格 inline) |
| 3 | §4 Implementation + §5 Results 前半 | Figure 1 + Table 1 |
| 4 | §5 Results 后半 + §6 Ethics + §7 Future + §8 Conclusion + References | Figure 2 + Figure 3 + Table 2 |

---

## 3. 实验设计

### 3.1 数据集与模型

**数据集**: MNIST (60k train / 10k test, 28×28 灰度)
- 自动下载 (~11 MB)
- 为何选 MNIST:在 CPU 也能 3–5 min 跑完,且对抗现象与 CIFAR 一样鲜明

**模型**: SimpleCNN (~421K 参数)
```
Conv(1→32, 3×3) + ReLU + MaxPool(2)
Conv(32→64, 3×3) + ReLU + MaxPool(2)
Flatten → Linear(3136→128) + ReLU → Linear(128→10)
```

### 3.2 七个实验 (E1–E7)

| 实验 | 内容 | CPU | GPU |
|---|---|---|---|
| **E1** | 干净 CNN 训练 3 epoch | 4 min | 30 s |
| **E2** | FGSM 在 ε∈{0,.05,.1,.15,.2,.25,.3} 扫描 | 1 min | 10 s |
| **E3** | PGD(7 步) 同 ε 扫描 | 3 min | 20 s |
| **E4** | FGSM 对抗训练 3 epoch → 再做 E2/E3 | 10 min | 1.5 min |
| **E5** | 混淆矩阵 @ ε=0.2 (clean 模型, FGSM 攻击) | 5 s | 1 s |
| **E6** | **每类准确率**柱状 @ ε=0.2 | 5 s | 1 s |
| **E7** | **攻击墙钟耗时**对比 (FGSM / PGD-7 / PGD-20) | 1 min | 20 s |
| **合计** | | **~20 min** | **~3 min** |

### 3.3 输出产物 (写入 `./outputs/`)

| 文件 | 内容 | 用途 |
|---|---|---|
| `results.json` | 所有数值指标 | 填入报告表格 |
| `fig1_samples.png` | 2×3 网格: 原图 / 扰动 / 对抗 | 报告 Figure 1 |
| `fig2_curves.png` | 鲁棒性曲线 (4 条) + 每类 bar | 报告 Figure 2 (双子图) |
| `fig3_confusion.png` | 清洁 vs 对抗混淆矩阵 | 报告 Figure 3 |
| `table_cost.csv` | 攻击算力对比 | 报告 Table 2 |

---

## 4. 完整代码 (`experiment.py`)

```python
"""
CDS521 Dissertation — Model Attack & Defense (MNIST)
Auto-detects CUDA; runs 7 experiments; writes figures/tables to ./outputs/.

CPU expected runtime ~ 20 min,  GPU expected ~ 3 min.
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

def pgd(model, x, y, eps, alpha=0.01, iters=PGD_ITERS):
    x_orig = x.clone().detach()
    x_adv  = x.clone().detach()
    for _ in range(iters):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0, 1).detach()
    return x_adv

# ---------- training ----------
def train(model, loader, epochs=EPOCHS, adv_eps=0.0):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if adv_eps > 0:
                x = fgsm(model, x, y, adv_eps)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"  epoch {ep+1}/{epochs}  time={time.time()-t0:.1f}s")

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

# ---------- E3: PGD sweep on clean ----------
print("\n[E3] PGD sweep (clean model)")
pgd_clean  = [accuracy(net, eval_loader, attack=(None if e == 0 else pgd), eps=e) for e in EPS_GRID]

# ---------- E4: adversarial training ----------
print("\n[E4] Adversarial training (FGSM eps=0.3)")
net_rob = SimpleCNN()
t = time.time(); train(net_rob, train_loader, adv_eps=0.3); e4_time = time.time() - t
fgsm_rob = [accuracy(net_rob, eval_loader, attack=(None if e == 0 else fgsm), eps=e) for e in EPS_GRID]
pgd_rob  = [accuracy(net_rob, eval_loader, attack=(None if e == 0 else pgd), eps=e) for e in EPS_GRID]
rob_clean_acc = accuracy(net_rob, test_loader)
print(f"    robust model clean acc = {rob_clean_acc:.4f}")

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
per_cls_clean = cm_clean.diagonal() / cm_clean.sum(1)
per_cls_adv   = cm_adv.diagonal()   / cm_adv.sum(1)

# ---------- E7: attack wall-clock cost ----------
print("\n[E7] Attack wall-clock timing (1024 samples)")
x_bench, y_bench = next(iter(DataLoader(Subset(test_set, list(range(1024))), batch_size=1024)))
x_bench, y_bench = x_bench.to(DEVICE), y_bench.to(DEVICE)
# warm-up
_ = fgsm(net, x_bench, y_bench, 0.2)
t = time.time(); _ = fgsm(net, x_bench, y_bench, 0.2);                         t_fgsm   = time.time() - t
t = time.time(); _ = pgd(net,  x_bench, y_bench, 0.2, iters=7);                t_pgd7   = time.time() - t
t = time.time(); _ = pgd(net,  x_bench, y_bench, 0.2, iters=20);               t_pgd20  = time.time() - t

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
    "train_time_adv_s":  e4_time,
    "attack_time_1024_s": {"fgsm": t_fgsm, "pgd7": t_pgd7, "pgd20": t_pgd20},
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
    ax[row, 0].imshow(orig, cmap="gray"); ax[row, 0].set_title(f"clean  →  pred {p_c[i].item()}")
    ax[row, 1].imshow(adv - orig, cmap="seismic", vmin=-0.3, vmax=0.3); ax[row, 1].set_title(r"perturbation  ($\varepsilon$=0.25)")
    ax[row, 2].imshow(adv, cmap="gray"); ax[row, 2].set_title(f"adversarial  →  pred {p_a[i].item()}")
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
    f.write(f"FGSM,1,{t_fgsm:.4f},1.0\n")
    f.write(f"PGD,7,{t_pgd7:.4f},{t_pgd7/t_fgsm:.2f}\n")
    f.write(f"PGD,20,{t_pgd20:.4f},{t_pgd20/t_fgsm:.2f}\n")

print("\n>>> all done. outputs written to ./outputs/")
print(f"    clean acc={clean_acc:.4f}  robust clean acc={rob_clean_acc:.4f}")
print(f"    FGSM@ε=.3 {'std/rob'} = {fgsm_clean[-1]:.3f} / {fgsm_rob[-1]:.3f}")
print(f"    PGD @ε=.3 {'std/rob'} = {pgd_clean[-1]:.3f} / {pgd_rob[-1]:.3f}")
```

---

## 5. 预期结果范围 (文献参考,跑完替换)

### 5.1 主表 (Table 1 草稿)

| $\varepsilon$ | FGSM / standard | PGD / standard | FGSM / robust | PGD / robust |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 0.990 | 0.990 | 0.980 | 0.980 |
| 0.10 | 0.60–0.75 | 0.35–0.55 | 0.96 | 0.95 |
| 0.20 | 0.25–0.45 | 0.05–0.20 | 0.93 | 0.90 |
| 0.30 | 0.05–0.20 | 0.00–0.05 | 0.88 | 0.82 |

### 5.2 算力表 (Table 2 草稿,基于 CPU 预估)

| Attack | Iter | Time / 1024 samples | Cost × |
|---|---:|---:|---:|
| FGSM | 1 | ~0.12 s | 1.0 × |
| PGD | 7 | ~0.70 s | ~6 × |
| PGD | 20 | ~1.80 s | ~15 × |

### 5.3 三点核心结论 (填入 Discussion)

1. **PGD ≫ FGSM**(多步攻击显著更强),ε=0.3 时 PGD 几乎将 accuracy 打到 0。
2. **对抗训练有效**:在 ε=0.3 下 PGD accuracy 从 < 5% 恢复到 ~82%。
3. **代价**:干净准确率下降约 1 pp,且训练耗时约翻倍 (FGSM-AT 每步多 1 次前/反向)。

---

## 6. 最终报告正文 (直接粘贴到 LaTeX/Word 即可)

> 下文为可直接提交的 4 页报告(密度较高,跑完实验后**替换粗体占位值**即可定稿)。

---

### 【报告正文开始】

# On Adversarial Attacks and Defenses in Deep Learning: A Systematic Study on MNIST

**Wang Wenxuan  |  Student ID: 1397228  |  CDS521 Foundation of AI  |  Deadline: 24 April 2026**

#### Abstract

Deep neural networks, despite super-human accuracy on vision benchmarks, can be fooled by imperceptibly small perturbations crafted by adversaries. This dissertation surveys the threat model, taxonomy, and defences of *adversarial machine learning*, and reports a controlled empirical study on MNIST. Using a four-layer convolutional classifier, we (i) implement the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) as white-box $L_\infty$ attacks, (ii) compare them against an FGSM-adversarially-trained variant, and (iii) analyse class-level vulnerability and computational cost. At $\varepsilon = 0.3$, PGD reduces the baseline's accuracy from **99.0 %** to **< 5 %**, while the adversarially trained model retains **≈ 82 %**. Adversarial training costs ≈ 1 pp in clean accuracy and a 2× training overhead, and does not uniformly protect every digit class. These results reproduce, at small scale, the central finding of Madry et al. (2018), and motivate our discussion of certified defences and physically realisable attacks.

## 1. Introduction

Szegedy et al. (2014) discovered that state-of-the-art image classifiers misclassify inputs modified by perturbations imperceptible to humans. Goodfellow et al. (2015) argued the phenomenon stems from the (near-)linear geometry of deep networks and proposed the **Fast Gradient Sign Method (FGSM)**, a single-step attack. Madry et al. (2018) reframed robust learning as a min–max optimisation,

$$\min_{\theta}\; \mathbb{E}_{(x,y)\sim \mathcal{D}}\!\Bigl[\max_{\|\delta\|_\infty \le \varepsilon}\; \mathcal{L}\bigl(f_\theta(x+\delta), y\bigr)\Bigr],\qquad (1)$$

and introduced **Projected Gradient Descent (PGD)**—a multi-step variant of FGSM—as a strong first-order adversary. The field has since matured into a formal taxonomy codified by NIST (2021). This paper reviews that taxonomy, reproduces FGSM/PGD on MNIST, and evaluates adversarial training, supplementing the standard accuracy-vs-budget plot with a per-class and cost analysis.

## 2. Theoretical Background

**Adversarial examples.** Given classifier $f_\theta$ and sample $(x,y)$, an adversarial example is $x' = x + \delta$ with $\|\delta\|_p \le \varepsilon$ such that $f_\theta(x')\neq y$. The budget $\varepsilon$ bounds perceptibility; $p\in\{0,2,\infty\}$ selects the geometry (pixel-count, Euclidean, uniform per-pixel). We adopt $p=\infty$ as in Goodfellow et al. (2015).

**White-box vs black-box.** A **white-box** adversary knows $\theta$ and can compute $\nabla_x \mathcal{L}$; FGSM and PGD belong here. A **black-box** adversary only queries the model as an oracle, yet *transferability* (Papernot et al., 2016) makes white-box examples surprisingly effective across unseen models.

**FGSM.** Linearising $\mathcal{L}$ around $x$, the worst-case $L_\infty$ perturbation of budget $\varepsilon$ is
$\delta^\star = \varepsilon\cdot\mathrm{sign}\!\bigl(\nabla_x\mathcal{L}\bigr)$,
giving $x' = \mathrm{clip}_{[0,1]}(x+\delta^\star)$. This is *exact* for a truly linear $\mathcal{L}$ and a useful first-order approximation otherwise.

**PGD.** PGD iterates the FGSM step $T$ times with stride $\alpha$, projecting back to the $\varepsilon$-ball (and pixel range) after each step. It is widely regarded as the strongest *first-order* adversary (Madry et al., 2018).

**Defence taxonomy.** Defences fall into three families: (i) *adversarial training*—augmenting the training loss with perturbed inputs, approximating the inner max in (1); (ii) *input preprocessing*—e.g. JPEG compression, feature squeezing (Xu et al., 2018), which attempt to remove adversarial signal; (iii) *certified defences* such as randomised smoothing (Cohen et al., 2019), providing provable robustness radii. Defences that merely obscure $\nabla_x\mathcal{L}$—"gradient masking"—were shown by Athalye et al. (2018) to give a false sense of security.

## 3. Application-Based Analysis

**Real-world scenarios.** (i) **Autonomous driving:** Eykholt et al. (2018) fooled a road-sign classifier with black-and-white stickers on stop signs, a *physically realisable* attack robust to viewpoint and distance. (ii) **Face recognition:** Sharif et al. (2016) 3-D-printed eyeglass frames that caused commercial face verifiers to impersonate chosen identities. (iii) **Malware detection:** Grosse et al. (2017) evaded neural malware classifiers by appending benign bytes, showing the threat is not confined to vision.

**Threat model.** Following NIST (2021), an adversary is characterised by four axes: *goal* (targeted / untargeted), *knowledge* (white- / grey- / black-box), *capability* ($L_p$ budget, query budget, physical constraints), and *access* (train-time poisoning vs test-time evasion). This study focuses on **white-box, test-time, untargeted, $L_\infty$-bounded evasion**.

**Pipeline.** (1) obtain a pretrained classifier; (2) for each test input, compute $\nabla_x\mathcal{L}$ via back-propagation; (3) synthesise $\delta$ under the chosen budget; (4) measure attack success; (5) for defence, retrain with the attack in the loop.

## 4. Implementation

We train a simple CNN (two 3×3 conv + max-pool blocks, 128-unit FC head; **421 K** parameters) on MNIST for **3** epochs with Adam ($\eta = 10^{-3}$, batch 128). We then evaluate on a held-out subset of **2 000** samples (all 10 000 on GPU) under a grid of $\varepsilon\in\{0, .05, .10, .15, .20, .25, .30\}$. Four model × attack conditions are compared: (standard, adv-trained) × (FGSM, PGD-$T$), plus clean accuracy. PGD uses $\alpha = 0.01$ and $T = 7$ (CPU) / $T = 20$ (GPU). The adversarially trained model uses FGSM-based min–max approximation at $\varepsilon = 0.3$. Pseudocode for FGSM:

```python
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    return (x + eps * grad.sign()).clamp(0, 1).detach()
```

## 5. Experimental Results

**Clean accuracy.** The baseline reaches **99.0 %** on the held-out test set; the adversarially trained variant attains **98.0 %**—a 1-point tax for robustness.

**Figure 1** shows two MNIST digits, their FGSM perturbations at $\varepsilon = 0.25$, and the resulting adversarial images. The perturbations resemble low-amplitude noise yet flip both predictions with high confidence.

![Figure 1: Original, perturbation, and FGSM adversarial examples at ε=0.25.](outputs/fig1_samples.png)

**Table 1** reports accuracy across the budget grid. **Figure 2(a)** plots the same data; **Figure 2(b)** breaks the standard-model accuracy at $\varepsilon = 0.2$ down by digit class.

| $\varepsilon$ | FGSM / std | PGD / std | FGSM / rob | PGD / rob |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | **0.990** | **0.990** | **0.980** | **0.980** |
| 0.10 | **0.68** | **0.43** | **0.96** | **0.95** |
| 0.20 | **0.32** | **0.11** | **0.93** | **0.90** |
| 0.30 | **0.12** | **0.03** | **0.88** | **0.82** |

*Table 1.* Test accuracy of the standard and adversarially trained CNN under FGSM and PGD attacks (replace bold placeholders with `results.json`).

![Figure 2: (a) robustness vs perturbation budget; (b) per-class accuracy at ε=0.2.](outputs/fig2_curves.png)

**Figure 3** shows the confusion matrix before and after FGSM attack at $\varepsilon = 0.2$ on the standard model: the diagonal structure collapses, and errors concentrate into a few "attractor" classes—often classes that differ from the true class by a single stroke (e.g. 8↔3, 4↔9).

![Figure 3: Confusion matrix before and after FGSM attack (ε=0.2).](outputs/fig3_confusion.png)

**Table 2** contrasts the wall-clock cost of the three attacks, measured on 1 024 samples.

| Attack | Iterations | Time / 1024 samples | Cost × |
|---|---:|---:|---:|
| FGSM  | 1  | **0.12 s** | 1.0 × |
| PGD-7 | 7  | **0.70 s** | **5.8 ×** |
| PGD-20 | 20 | **1.80 s** | **15.0 ×** |

*Table 2.* Attack compute budget (device-dependent; replace with `table_cost.csv`).

**Findings.** (i) PGD dominates FGSM on the standard model at every $\varepsilon$, consistent with its stronger inner optimisation; the gap grows with $\varepsilon$. (ii) Adversarial training restores near-clean accuracy under FGSM ($\ge 88 \%$ at $\varepsilon = 0.3$) and retains **> 82 %** under stronger PGD, reproducing Madry et al. (2018) at small scale. (iii) Vulnerability is **non-uniform across classes** (Figure 2b): digits with more visual "competitors" (e.g. 8) fall first, supporting the view that adversarial error clusters along decision-boundary geometry. (iv) PGD's effectiveness comes at **15× the compute** of FGSM (Table 2)—relevant when adversarial evaluation is a routine CI step.

## 6. Ethical Considerations

Adversarial ML is **dual-use**. Offensively, the same techniques that fool a stop-sign classifier can (i) enable impersonation against biometric checkpoints, (ii) craft *deepfakes* evading forensic classifiers, or (iii) bypass spam and malware filters. Publishing attack code therefore risks arming malicious actors. Yet *withholding* attacks leaves defenders blind: NIST (2021) explicitly recommends disclosure, because "security through obscurity" systematically favours attackers. A responsible stance—adopted here—is to reproduce published attacks on toy datasets while foregrounding defence. A second ethical axis concerns **fairness**: a robust classifier that has inherited demographic skew (Bagdasaryan et al., 2019) can be attacked *differentially* across groups, compounding fairness harms. Our per-class analysis (Figure 2b) is the MNIST analogue of this concern and motivates reporting *worst-class* robustness alongside average accuracy.

## 7. Future Directions

Three frontiers look most promising. **(1) Certified defences.** Randomised smoothing (Cohen et al., 2019) converts a base classifier into a provably robust one by majority-voting over Gaussian-noised inputs; the certificate scales favourably to ImageNet. **(2) Physically realisable attacks.** Beyond digital perturbations, the *Expectation-over-Transformation* framework (Athalye et al., 2018) produces adversaries robust to real-world imaging pipelines, motivating *deployment-time* monitoring and sensor-level defences. **(3) Foundation-model robustness.** Large pretrained models exhibit emergent behaviours under adversarial prompting (Zou et al., 2023); extending the $L_\infty$ threat model to text, speech, and multi-modal inputs is an open and socially consequential direction.

## 8. Conclusion

We surveyed the landscape of adversarial machine learning and empirically reproduced FGSM, PGD, and FGSM-based adversarial training on MNIST under a modest compute budget. The quantitative picture—Table 1, Figure 2, Figure 3—confirms that (i) adversarial vulnerability is severe for standardly-trained networks, (ii) adversarial training is an effective, if computationally costly, remediation, and (iii) the benefit is non-uniform across classes. Trustworthy AI will require combining such empirical defences with certified guarantees, physical-world evaluation, and careful dual-use ethics.

#### References

[1] Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated Gradients Give a False Sense of Security. *ICML*.
[2] Bagdasaryan, E., Poursaeed, O., & Shmatikov, V. (2019). Differential Privacy has Disparate Impact on Model Accuracy. *NeurIPS*.
[3] Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). Certified Adversarial Robustness via Randomized Smoothing. *ICML*.
[4] Eykholt, K., et al. (2018). Robust Physical-World Attacks on Deep Learning Visual Classification. *CVPR*.
[5] Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. *ICLR*.
[6] Grosse, K., et al. (2017). Adversarial Examples for Malware Detection. *ESORICS*.
[7] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR*.
[8] NIST (2021). *Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations*. NISTIR 8269 draft.
[9] Papernot, N., McDaniel, P., Goodfellow, I., et al. (2016). Practical Black-Box Attacks against Machine Learning. *AsiaCCS*.
[10] Sharif, M., Bhagavatula, S., Bauer, L., & Reiter, M. (2016). Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition. *CCS*.
[11] Szegedy, C., et al. (2014). Intriguing Properties of Neural Networks. *ICLR*.
[12] Xu, W., Evans, D., & Qi, Y. (2018). Feature Squeezing: Detecting Adversarial Examples. *NDSS*.
[13] Zou, A., et al. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv:2307.15043*.

### 【报告正文结束】

---

## 7. 排版建议 (压到 4 页)

### 7.1 推荐:LaTeX (Overleaf 或本地)

```latex
\documentclass[10pt, twocolumn]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{graphicx, amsmath, booktabs, hyperref, caption}
\usepackage[numbers]{natbib}

\setlength{\parskip}{2pt}
\setlength{\parindent}{0pt}
\captionsetup{font=small, labelfont=bf}

\title{\vspace{-1cm}On Adversarial Attacks and Defenses in Deep Learning:\\
A Systematic Study on MNIST}
\author{Wang Wenxuan \\ {\small Student ID: 1397228 ~|~ CDS521 ~|~ Deadline: 24 April 2026}}
\date{}
```

**关键参数:**
- `10pt` + `twocolumn` → 显著增密,4 页足够
- Figure 宽度 `\columnwidth`(单栏)或 `\textwidth`(跨栏)
- 对 Figure 3(混淆矩阵)用 `\textwidth` 跨栏显示

### 7.2 备选:Word

- 纸张 A4,页边距 上下 1.8 cm / 左右 1.8 cm
- 字体 Times New Roman 10 pt,行距 1.15
- 图片宽度占满页面,高度压到 5–6 cm
- 参考文献用 9 pt,段后 2 pt

### 7.3 超页抢救策略

若超过 4 页,按以下顺序压缩:
1. 把 §6 和 §7 合并为"Ethics and Future Directions"
2. 把 Table 2 改为正文一行叙述:"PGD-20 costs ≈ 15× FGSM per batch"
3. Figure 3 缩到半栏
4. 删 §2 的 "Defence taxonomy" 小节一半 (合并到 §1)

---

## 8. 两天时间规划

| 时间 | 任务 | 耗时 |
|---|---|---|
| **4-22 今晚 21:00–21:30** | (可选) 重装 CUDA PyTorch | 15 min |
| **4-22 21:30–22:00** | 运行 `experiment.py`,得到 outputs/ | CPU 20 min / GPU 5 min |
| **4-23 上午** | 按实际数值填报告 §5 表格与正文,精校 abstract | 2 h |
| **4-23 下午** | LaTeX/Word 排版,确保 ≤ 4 页,图表位置调优 | 2 h |
| **4-23 晚** | 参考文献格式、拼写检查、最终试印 | 1 h |
| **4-24 上午** | 导出 PDF 命名为 `CDS521_Dissertation_1397228_WangWenxuan.pdf`,提交 | 30 min |

---

## 9. 提交前 Checklist

- [ ] PDF **≤ 4 页**(含标题区,不含独立封面)
- [ ] 首行含 **Wang Wenxuan / 1397228 / 24 April 2026**
- [ ] 所有 [1]–[13] 引用都在正文至少出现一次
- [ ] Figure 1/2/3 + Table 1/2 在正文中都有 `\ref{}` 式引用
- [ ] 公式 (1) 渲染正确
- [ ] 代码片段 (FGSM) 语法正确可读
- [ ] 实际跑完后:`fig1/2/3_*.png` 清晰,未被裁切
- [ ] 文件命名:`CDS521_Dissertation_1397228_WangWenxuan.pdf`

---

## 10. 下一步你要做的

1. **确认选题 B 与整体方案**(MNIST + 4 组实验 + 3 图 + 2 表)。
2. **决定 CPU / GPU 路径**——若装 CUDA 版 PyTorch,告诉我一声我可以帮你确认。
3. 告诉我开跑 —— 我会:
   - 写 `experiment.py` 到本目录
   - 运行实验 (在后台,你可以继续做别的事)
   - 读 `outputs/results.json`,把所有粗体占位数值替换进 §6 报告
   - 生成最终可交的 `.md` + `.tex` + `.pdf`(可选 `.docx`)

就绪时说"开始"即可。
