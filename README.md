# Diffusion Models on Graphs with HydraGNN 
Welcome to our Diffusion Model repository! 🎉 This project builds on HydraGNN, leveraging its powerful GNN and ML utilities for training, testing, and model optimization. We've added diffusion modeling on graphs to explore new horizons in graph-based learning. 🚀

## Features
    * Graph Neural Networks (GNN): Utilizing HydraGNN's robust GNN architecture for effective graph modeling. 
    * Diffusion Modeling: Enhanced with diffusion processes to capture complex graph dynamics. 
    * Parallelization & Optimization: Effortlessly scale with multi-GPU support and optimized training routines! ⚡

## Quick Start
Clone the repo:

```bash
git clone https://github.com/your-repo/diffusion-gnn.git
cd diffusion-gnn
```

### Install Dependencies:
Make sure you have the HydraGNN environment set up:
```bash
pip install -r requirements.txt
```

### Run Training:
```bash
python train.py --config config.yaml
```

## How It Works
HydraGNN integration 🌊: We utilize the operational utilities from HydraGNN, such as model training, testing, and optimization, to simplify workflow.
Diffusion Process: Modeled on graph structures to simulate the propagation of information or features across the graph nodes. Perfect for dynamic systems! 
Model Parallelization: Thanks to HydraGNN, training large models with multi-GPU support is integrated.

### ️Configuration
All model and training parameters can be easily set via our config.json file:

```json
model:
  type: diffusion_gnn
  layers: 5
  hidden_dim: 128
train:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## Modules
`src/....py: `

### Performance
Our diffusion-enhanced GNNs show promising results in tasks such as:

### Contributing
We welcome contributions! If you're interested in extending the diffusion model or improving performance, feel free to submit a pull request or open an issue. 
