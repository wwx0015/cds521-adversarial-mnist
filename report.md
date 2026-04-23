# On Adversarial Attacks and Defenses in Deep Learning: A Systematic Study on MNIST

**Wang Wenxuan  |  Student ID: 1397228  |  CDS521 Foundation of AI  |  Deadline: 24 April 2026**

Code and data: <https://github.com/wwx0015/cds521-adversarial-mnist>

> Markdown mirror of `report.tex` / `CDS521_Dissertation_1397228_WangWenxuan.pdf`. All numerical values are the actual experimental output from `experiment.py` (see `outputs/results.json`).

## Abstract

Deep neural networks, despite high accuracy on many vision benchmarks, can be fooled by imperceptibly small perturbations crafted by adversaries. This dissertation surveys the threat model, taxonomy, and defenses of *adversarial machine learning*, and reports a controlled empirical study on MNIST. Using a four-layer convolutional classifier (421 K parameters) trained on an NVIDIA RTX 2050 GPU, we (i) implement the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) as white-box $L_\infty$ attacks, (ii) compare them against a Madry-style PGD adversarially trained variant, and (iii) analyze class-level vulnerability and compute cost. At $\varepsilon = 0.3$, PGD-20 reduces the baseline's accuracy from **98.86 %** to effectively **0 %**, whereas the adversarially trained model retains **79.00 %**. Adversarial training costs ≈ 5.2 pp in clean accuracy and a 6.6× training wall-clock overhead, and robustness is demonstrably non-uniform across digit classes. These results are consistent with the central finding of Madry et al. (2018) at small scale.

## 1. Introduction

Szegedy et al. [11] discovered that state-of-the-art image classifiers misclassify inputs perturbed by modifications imperceptible to humans. Goodfellow et al. [5] argued the phenomenon stems from the (near-)linear geometry of deep networks and proposed the **Fast Gradient Sign Method (FGSM)**, a single-step attack. Madry et al. [7] reframed robust learning as a min–max optimization

$$\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\max_{\|\delta\|_\infty\le\varepsilon} \mathcal{L}(f_\theta(x+\delta),y)\right], \qquad (1)$$

and introduced **Projected Gradient Descent (PGD)**—a multi-step variant of FGSM—as a strong first-order adversary. The field has since matured into a formal taxonomy codified by NIST [8]. This paper reviews that taxonomy, reproduces FGSM and PGD on MNIST, evaluates PGD-based adversarial training, and supplements the standard accuracy-vs-budget plot with class-level and computational-cost analyses.

## 2. Theoretical Background

**Adversarial examples.** Given a classifier $f_\theta$ and a sample $(x,y)$, an adversarial example is $x'=x+\delta$ with $\|\delta\|_p\le\varepsilon$ such that $f_\theta(x')\neq y$. The budget $\varepsilon$ bounds perceptibility; $p\in\{0,2,\infty\}$ selects the geometry. We adopt $p=\infty$.

**White-box vs. black-box.** A white-box adversary knows $\theta$ and can compute $\nabla_x\mathcal{L}$; FGSM and PGD belong here. A black-box adversary only queries the model as an oracle, yet transferability [9] makes white-box examples surprisingly effective across unseen models.

**FGSM.** Linearizing $\mathcal{L}$ around $x$, the worst-case $L_\infty$ perturbation of budget $\varepsilon$ is $\delta^\star=\varepsilon\,\text{sign}(\nabla_x\mathcal{L})$, giving $x'=\text{clip}_{[0,1]}(x+\delta^\star)$.

**PGD.** PGD iterates the FGSM step $T$ times with stride $\alpha$, projecting back to the $\varepsilon$-ball after each step. It is widely regarded as the strongest first-order adversary [7].

**Defense taxonomy.** Three families: (i) adversarial training—augmenting the loss with perturbed inputs, approximating the inner max in (1); (ii) input preprocessing such as feature squeezing [12]; and (iii) certified defenses such as randomized smoothing [3], which provide provable robustness radii. Defenses that merely obscure $\nabla_x\mathcal{L}$—"gradient masking"—were shown by Athalye et al. [1] to give a false sense of security.

## 3. Application-Based Analysis

**Real-world scenarios.** (i) *Autonomous driving:* Eykholt et al. [4] fooled a road-sign classifier with black-and-white stickers on stop signs—a physically realizable attack. (ii) *Face recognition:* Sharif et al. [10] 3-D-printed eyeglass frames that caused commercial face verifiers to impersonate chosen identities. (iii) *Malware detection:* Grosse et al. [6] evaded neural malware classifiers by appending benign bytes.

**Threat model.** Following NIST [8], an adversary is characterized by four axes: *goal* (targeted/untargeted), *knowledge* (white-/grey-/black-box), *capability* ($L_p$ budget, query budget, physical constraints), *access* (train-time poisoning vs. test-time evasion). This study focuses on **white-box, test-time, untargeted, $L_\infty$-bounded evasion**.

**Pipeline.** (1) train or obtain a target classifier; (2) compute $\nabla_x\mathcal{L}$ via back-propagation; (3) synthesize $\delta$ under the chosen budget; (4) measure attack success; (5) for defense, retrain with the attack in the loop, approximating Eq. (1).

## 4. Implementation

We train a CNN (two 3×3 conv + max-pool blocks, 128-unit FC head; 421 K parameters) on MNIST for 3 epochs with Adam ($\eta=10^{-3}$, batch 128). The full 10 000-sample test set is used for evaluation. Seven $\varepsilon$ values are swept, $\varepsilon\in\{0,.05,.10,.15,.20,.25,.30\}$. Four model × attack conditions are compared: (standard, adv-trained) × (FGSM, PGD-20). Evaluation PGD follows the Madry protocol: $T=20$ steps, random start inside the $\varepsilon$-ball, and step size $\alpha=2.5\varepsilon/T$, so the attack can reach the full ball boundary at every $\varepsilon$. The adversarially trained model is obtained via PGD-7 with random initialization, $\alpha=\varepsilon/4$, $\varepsilon=0.3$, for 5 epochs (Madry-style). All experiments run on an NVIDIA GeForce RTX 2050 (4 GB).

```python
def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    return (x + eps * grad.sign()).clamp(0, 1).detach()
```

## 5. Experimental Results

**Clean accuracy.** The baseline reaches **98.86 %**; the PGD adversarially trained variant attains **93.71 %**—a 5.15 pp tax for robustness.

**Figure 1** shows two MNIST digits, their FGSM perturbations at $\varepsilon=0.25$, and the resulting adversarial images. Both predictions flip (7→3 and 2→6).

![Figure 1](outputs/fig1_samples.png)

**Table 1.** Test accuracy (%) of the standard and PGD adversarially trained CNN under FGSM and PGD-20 across the $L_\infty$ budget grid. PGD saturates at 0 % on the standard model by $\varepsilon=0.25$.

| $\varepsilon$ | FGSM / std | PGD / std | FGSM / rob | PGD / rob |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 98.86 | 98.86 | 93.71 | 93.71 |
| 0.05 | 95.36 | 93.84 | 92.09 | 91.92 |
| 0.10 | 84.47 | 69.48 | 90.47 | 89.92 |
| 0.15 | 60.38 | 18.84 | 89.00 | 87.97 |
| 0.20 | 31.71 |  0.47 | 87.80 | 85.54 |
| 0.25 | 11.19 |  0.00 | 86.92 | 82.89 |
| 0.30 |  2.56 |  0.00 | 86.13 | 79.00 |

**Figure 2.** (a) Robustness vs. perturbation budget $\varepsilon$. (b) Per-class accuracy at $\varepsilon=0.2$ for the standard model.

![Figure 2](outputs/fig2_curves.png)

**Figure 3.** Confusion matrices before (blue) and after (red) FGSM at $\varepsilon=0.2$ on the standard model. Errors concentrate into attractor columns rather than scattering uniformly.

![Figure 3](outputs/fig3_confusion.png)

**Table 2.** Wall-clock cost per 1024 images (NVIDIA RTX 2050).

| Attack | Iterations | Time (s) | Relative |
|---|---:|---:|---:|
| FGSM  | 1  | 0.047 |  1.00× |
| PGD   | 7  | 0.304 |  6.43× |
| PGD   | 20 | 0.829 | 17.50× |

**Findings.**
1. PGD dominates FGSM on the standard model at every $\varepsilon$; the gap grows from 1.52 to 41.54 pp between $\varepsilon=0.05$ and 0.15 and closes only because PGD saturates at 0 % by $\varepsilon=0.25$.
2. Adversarial training retains 79.00 % accuracy under PGD-20 at $\varepsilon=0.3$ while the standard model collapses to 0 %—a result consistent with the central finding of Madry et al. [7] at small scale.
3. Robustness is **non-uniform across classes** (Figure 2b): at $\varepsilon=0.2$, digit 1 retains 69.5 % while digit 9 falls to 13.0 %—a 56.5 pp spread.
4. PGD-20's 17.5× cost over FGSM matters when adversarial evaluation is a routine CI step.
5. Adversarial training took 112.8 s vs 17.0 s for clean training, a 6.6× total overhead (≈4× per-epoch × 1.67× longer schedule).

## 6. Ethical Considerations

Adversarial ML is **dual-use**. Offensively, the same techniques that fool a stop-sign classifier can (i) enable impersonation against biometric checkpoints, (ii) craft deepfakes evading forensic classifiers, or (iii) bypass spam and malware filters. Yet *withholding* attacks leaves defenders blind: NIST [8] explicitly recommends disclosure because "security through obscurity" systematically favors attackers. A responsible stance—adopted here—reproduces published attacks on toy datasets while foregrounding defense. A second ethical axis concerns **fairness**: reporting only *average* robustness can hide the fact that some classes or subgroups are markedly more vulnerable than others, so fairness harms may compound under adversarial attack [8]. Our per-class analysis (Figure 2b) is the MNIST analogue of this concern—robustness gaps spanning 56.5 pp between the easiest and hardest digit at $\varepsilon=0.2$—motivating the reporting of *worst-class* robustness alongside average accuracy.

## 7. Future Directions

**(1) Certified defenses.** Cohen et al. [3] convert a base classifier into a provably robust one by majority-voting over Gaussian-noised inputs; the certificate scales favorably to ImageNet. **(2) Physically realizable attacks.** The Expectation-over-Transformation framework of Athalye et al. [2] produces adversaries robust to real-world imaging pipelines, motivating sensor-level and deployment-time defenses. **(3) Foundation-model robustness.** Large pretrained models exhibit emergent behaviors under adversarial prompting [13]; extending the $L_\infty$ threat model to text, speech, and multi-modal inputs is an open and socially consequential direction.

## 8. Conclusion

We surveyed the landscape of adversarial machine learning and empirically reproduced FGSM, PGD, and Madry-style adversarial training on MNIST under a modest compute budget. The quantitative picture confirms that (i) adversarial vulnerability is severe for standardly trained networks, (ii) adversarial training is an effective, if costly, remediation, (iii) the benefit is non-uniform across classes, and (iv) strong attacks are 17.5× more expensive than weak ones. Trustworthy AI will require combining such empirical defenses with certified guarantees, physical-world evaluation, and careful dual-use ethics.

## References

[1] A. Athalye, N. Carlini, D. Wagner. Obfuscated gradients give a false sense of security. *ICML*, 2018.
[2] A. Athalye, L. Engstrom, A. Ilyas, K. Kwok. Synthesizing robust adversarial examples. *ICML*, 2018.
[3] J. Cohen, E. Rosenfeld, Z. Kolter. Certified adversarial robustness via randomized smoothing. *ICML*, 2019.
[4] K. Eykholt et al. Robust physical-world attacks on deep learning visual classification. *CVPR*, 2018.
[5] I. J. Goodfellow, J. Shlens, C. Szegedy. Explaining and harnessing adversarial examples. *ICLR*, 2015.
[6] K. Grosse et al. Adversarial examples for malware detection. *ESORICS*, 2017.
[7] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. Towards deep learning models resistant to adversarial attacks. *ICLR*, 2018.
[8] A. Vassilev, A. Oprea, A. Fordyce, H. Anderson. *Adversarial ML: A Taxonomy and Terminology of Attacks and Mitigations*, NIST AI 100-2e2023, 2024.
[9] N. Papernot, P. McDaniel, I. Goodfellow, et al. Practical black-box attacks against machine learning. *AsiaCCS*, 2016.
[10] M. Sharif, S. Bhagavatula, L. Bauer, M. Reiter. Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition. *CCS*, 2016.
[11] C. Szegedy et al. Intriguing properties of neural networks. *ICLR*, 2014.
[12] W. Xu, D. Evans, Y. Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. *NDSS*, 2018.
[13] A. Zou et al. Universal and transferable adversarial attacks on aligned language models. *arXiv:2307.15043*, 2023.
