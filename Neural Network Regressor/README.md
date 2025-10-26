# Neural Network Regressor

This project implements a **Feedforward Neural Network** in **PyTorch** for a **synthetic regression problem**.  
It explores model design, training procedures, evaluation, and experimental variations.

---

## Dataset
- **Samples:** 1000  
- **Features:** 1 input feature  
- **Preprocessing:** Standardized using `StandardScaler` (zero mean, unit variance)  
- **Split:** 80% training (800 samples), 20% testing (200 samples)

---

## Model Architecture
| Layer | Neurons | Activation |
|--------|----------|-------------|
| Input | 1 | — |
| Hidden Layer 1 | 20 | ReLU |
| Hidden Layer 2 | 10 | ReLU |
| Output | 1 | — |

> This simple yet deep-enough structure captures nonlinear relationships within the dataset.

---

## Training Details
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam (`lr = 0.001`)  
- **Batching:** Full-batch gradient descent  
- **Epochs:** 500  

**Training Progress:**  
Loss decreased consistently from **~353 → ~101**, showing smooth convergence.

---

## Model Evaluation
| Metric | Test Score |
|--------|-------------|
| MSE | 113.48 |
| MAE | 8.31 |
| R² | 0.70 |

> The model explains about 70% of the variance in unseen data — a strong indication of generalization and stability.

---

## Visualizations
- **Training Loss Curve:** Smooth, consistent decline  
- **Predicted vs Actual Scatter Plot:** Tight clustering around the diagonal → strong fit  

---

## Additional Experiments
| Experiment | Outcome | Observation |
|-------------|----------|--------------|
| **L2 Regularization (0.01)** | MSE ↑ to ~150.75, R² ↓ to ~0.61 | Slight over-regularization |
| **StepLR Scheduler** | MSE ↑ to ~356.85, R² ↓ to ~0.07 | Learning rate changes too aggressive |
| **Mini-batch (32)** | Similar to full-batch | No significant difference |
| **Increased Epochs (300 → 500)** | MSE ↓ to 121.69, R² ↑ to 0.68 | Longer training improved fit |

---

## Key Takeaways
- Successfully trained a **two-hidden-layer regressor** on a small dataset.  
- Demonstrated strong predictive power and generalization.  
- Explored **regularization**, **scheduling**, **batching**, and **training duration** systematically.  
- Gained insight into **neural network behavior** for regression problems.

---

## Tech Stack
- **Language:** Python  
- **Framework:** PyTorch  
- **Libraries:** NumPy, Scikit-learn, Matplotlib  


# Train and evaluate the model
python nn_regressor.py
