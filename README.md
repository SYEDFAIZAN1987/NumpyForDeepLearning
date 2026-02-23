🧠 NumPy for Deep Learning

Mastering the mathematical engine behind modern AI

This repository is a comprehensive guide to using NumPy specifically for Deep Learning applications — from tensor manipulation to implementing core functions like Softmax, ReLU, and Gumbel-Max.

🚀 Overview

Deep Learning is essentially:

Linear Algebra + Calculus + Code

While high-level frameworks like PyTorch and TensorFlow are industry standards, understanding the underlying NumPy implementations is crucial for:

Debugging neural networks

Building custom layers

Research experimentation

Strengthening mathematical intuition

📌 Key Concepts Covered
🔹 Tensor Surgery

Advanced slicing

np.stack vs np.concatenate

Axis manipulation

🔹 Activation Functions

Vectorized implementation of:

ReLU

Leaky ReLU

Softmax

🔹 Data Augmentation

np.flip

np.roll

np.pad

🔹 Stochastic Tricks

Implementing the Gumbel-Max Trick for differentiable sampling

🔹 Normalization

Calculating mean and variance across specific axes using keepdims=True

🛠️ Installation & Setup

To run the notebooks locally:

git clone https://github.com/SYEDFAIZAN1987/NumpyForDeepLearning.git
cd NumpyForDeepLearning
pip install numpy jupyterlab
📖 Deep Dive: Core Implementations
1️⃣ The Softmax Function

Essential for multi-class classification, implemented using the axis=-1 and keepdims=True pattern to support batch processing.

𝜎
(
𝑧
)
𝑖
=
𝑒
𝑧
𝑖
∑
𝑗
=
1
𝐾
𝑒
𝑧
𝑗
σ(z)
i
	​

=
∑
j=1
K
	​

e
z
j
	​

e
z
i
	​

	​

2️⃣ ReLU Activation (Non-Linearity)

Implemented via:

np.maximum(0, x)

This effectively “deactivates” neurons receiving negative signals.

3️⃣ Gumbel-Max Sampling

A technique used in LLMs and Reinforcement Learning to sample from categorical distributions.

# Adding Gumbel noise to logits
noise = np.random.gumbel(0, 1, logits.shape)
sample = np.argmax(logits + noise)
📂 Repository Structure
├── NumpyNotes.ipynb      # Main workbook with code & explanations
├── README.md             # Documentation
└── .gitignore            # Prevents unnecessary files (e.g., .ipynb_checkpoints)
🤝 Contributing

Found a more efficient way to implement a layer? Open a PR!

Steps:

Fork the project

Create your feature branch

git checkout -b feature/AmazingFeature

Commit your changes

git commit -m "Add some AmazingFeature"

Push to the branch

git push origin feature/AmazingFeature

Open a Pull Request

👤 Author

Syed Faizan
GitHub: @SYEDFAIZAN1987
