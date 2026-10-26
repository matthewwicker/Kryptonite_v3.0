# Kryptonite-3.0

Coursework for Mathematics for Machine Learning (60037/70015) at Imperial College London (ImpCML 2026).

### General Objective

The `PaperAndInstructions` directory holds the mock paper *Kryptonite-3.0: The Final End to Machine Learning Hype?*, the coursework instructions and the marking scheme. The paper proposes a family of challenge datasets that it claims modern machine learning cannot solve, and a pretrained model, KryptoNet-40, that it claims cannot be adapted to them. Your task is to write a response paper that refutes these claims theoretically and empirically, including an explanation of LoRA and its computational cost (see section 1C of the instructions).

### Contents

| Path | What it is |
|---|---|
| `PaperAndInstructions/Kryptonite_3_0_Prompt.pdf` | The mock paper |
| `PaperAndInstructions/ImpCML 2026 Coursework Instructions.pdf` | What to submit and how |
| `PaperAndInstructions/MML_Coursework__ImpCML.pdf` | Marking scheme and formatting rules |
| `Datasets/kryptonite-{20,30,40}-X.npy`, `-y.npy` | Labelled training data for Kryptonite-20, -30 and -40 |
| `Datasets/kryptonite-40-adapt-X.npy`, `-y.npy` | Labelled data for adapting KryptoNet-40 |
| `Datasets/hidden-kryptonite-{20,30,40}-X.npy`, `hidden-kryptonite-40-adapt-X.npy` | Unlabelled hidden sets (10,000 rows each) on which your predictions are graded |
| `kryptonet40.pt`, `kryptonet.py` | KryptoNet-40 weights and model definition |
| `MockPaperExperiments.ipynb` | Code that reproduces the mock paper's figures |
| `check_submission.py` | Checks the format of your hidden-set predictions |
| `ImpCMLAuthorPack.zip` | LaTeX template for your report |

### KryptoNet-40

```python
import numpy as np
import torch
from kryptonet import load_kryptonet

model = load_kryptonet("kryptonet40.pt")
X = torch.from_numpy(np.load("Datasets/kryptonite-40-adapt-X.npy"))
predictions = (model(X).squeeze(1) > 0).long()
```

### Submitting Predictions

Put your predictions in a directory called `hiddenlabels` at the top level of your GitLab repository, as `y_predicted_20.npy`, `y_predicted_30.npy`, `y_predicted_40.npy` and `y_predicted_40_adapt.npy`. Each file must load into a binary vector of length 10,000, in the same row order as the matching hidden set. Run `python check_submission.py path/to/your/repository` before submitting.

### Formatting

`ImpCMLAuthorPack.zip` contains the LaTeX source for the formatting instructions; place these files in Overleaf to stay within the required format. Reports should be between 3 and 6 pages, not counting references.

### Requirements

Python 3 with `numpy`, `scipy`, `scikit-learn`, `torch` and `matplotlib`. Every experiment runs on a laptop CPU.
