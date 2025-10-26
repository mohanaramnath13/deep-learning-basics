# Neural Network Optimization — Lab 2

This project investigates how different **optimization algorithms** affect the training behavior and performance of a neural network.  
The same model architecture was trained using **SGD**, **SGD with Momentum**, **RMSprop**, and **Adam**, and their results were compared.

---

## 1. Objective
To understand the role of various optimizers in neural network training and compare their convergence speed, stability, and accuracy.

---

## 2. Theory
Optimization minimizes a model’s loss function through gradient-based weight updates.  
Different optimizers modify this process:

| Optimizer | Description |
|------------|-------------|
| **SGD** | Applies gradients directly with a fixed learning rate. |
| **SGD + Momentum** | Adds a velocity term to gradients, improving convergence stability. |
| **RMSprop** | Uses an exponentially decaying average of squared gradients for adaptive learning rates. |
| **Adam** | Combines Momentum and RMSprop, using both mean and variance of gradients. |

> Learning rate selection is critical: too low → slow learning; too high → unstable updates.

---

## 3. Dataset
- **Samples:** 1000  
- **Features:** 20 numerical features  
- **Target:** Binary (0 or 1)  
- **Preprocessing:** Data normalized  
- **Split:** 80% training, 20% testing  

---

## 4. Model Architecture
| Layer | Neurons | Activation |
|--------|----------|-------------|
| Input | 20 | — |
| Hidden Layer 1 | 64 | ReLU |
| Hidden Layer 2 | 32 | ReLU |
| Output | 2 (logits for binary classification) | — |

- **Loss Function:** CrossEntropyLoss  
- **Learning Rate:** 0.01  
- **Epochs:** 20  

---

## 5. Training Setup
The same model was trained separately with each optimizer using PyTorch.  
Metrics recorded per epoch:
- Training and testing **loss**
- Training and testing **accuracy**

---

## 6. Results
| Optimizer | Training Accuracy | Testing Accuracy | Remarks |
|------------|------------------|------------------|----------|
| **SGD** | ~78.5% | ~76.5% | Slow convergence |
| **SGD + Momentum** | ~96.1% | ~91.0% | Faster, smoother training |
| **RMSprop** | ~100% | ~95.0% | Very fast convergence |
| **Adam** | ~99.9% | ~94.5% | Fast, stable, consistent performance |

---

## 7. Observations
- Basic **SGD** was slow and required more epochs for convergence.  
- **Momentum** improved SGD’s speed and final accuracy significantly.  
- **RMSprop** converged the fastest with high stability.  
- **Adam** offered a good trade-off between speed and robustness, with strong generalization.

---

## 8. Conclusion
Different optimizers yield distinct training behaviors and performance levels.  
- **SGD** is simple but slow.  
- **Momentum** enhances convergence.  
- **RMSprop** and **Adam** adapt learning rates effectively, achieving superior accuracy.  
Adaptive optimizers like **Adam** are generally preferred for faster, more stable training.

---

## 9. Tech Stack
- **Language:** Python  
- **Framework:** PyTorch  
- **Libraries:** NumPy, Matplotlib, Scikit-learn  
