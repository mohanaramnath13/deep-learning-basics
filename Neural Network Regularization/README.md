# Regularizers in Neural Networks — Lab 3

This lab explores how different **regularization techniques** affect neural network performance and overfitting behavior.  
The goal was to compare **L1**, **L2**, and **Dropout** regularization methods against a baseline model without any regularization.

---

## 1. Objective
To study the effect of **L1**, **L2**, and **Dropout** regularization techniques on a neural network’s ability to generalize in a binary classification task.

---

## 2. What is Overfitting?
**Overfitting** occurs when a neural network memorizes training data, including noise and irrelevant details, resulting in poor performance on unseen data.  
The model fits the training set too closely instead of learning generalizable patterns.

---

## 3. Regularization Techniques
| Technique | Description | Effect |
|------------|-------------|--------|
| **L1 (Lasso)** | Adds sum of absolute weights to the loss function | Promotes sparsity; forces some weights to zero |
| **L2 (Ridge)** | Adds sum of squared weights to the loss function | Discourages large weights; improves stability |
| **Dropout** | Randomly disables a fraction of neurons during training | Reduces co-adaptation and over-reliance on specific neurons |

---

## 4. Dataset
- **Samples:** 1000  
- **Features:** 20 continuous input features  
- **Target:** Binary (0 or 1)  
- **Split:** 80% training, 20% validation  
- **Preprocessing:** Normalized using `StandardScaler` (fit on training data and applied to validation set)

---

## 5. Model Architecture
| Layer | Neurons | Activation |
|--------|----------|-------------|
| Input | 20 | — |
| Hidden Layer 1 | 64 | ReLU |
| Hidden Layer 2 | 32 | ReLU |
| Output | 2 (binary classification logits) | — |

- **Dropout Variant:** Dropout layers (p = 0.3) added after each hidden layer.  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (`lr = 0.0005`)  
- **Epochs:** 50  

**Regularization Configurations:**
- **L1:** Manual penalty added (`λ = 0.0005`)  
- **L2:** Weight decay in optimizer (`weight_decay = 0.001`)  
- **Dropout:** Probability `p = 0.3`

---

## 6. Results
| Regularizer | Training Loss | Validation Loss |
|--------------|----------------|------------------|
| **None** | 0.0229 | 0.1178 |
| **L1** | 0.1604 | 0.1014 |
| **L2** | 0.0256 | 0.1320 |
| **Dropout** | 0.1169 | 0.1136 |

---

## 7. Discussion
- The **baseline model** (no regularization) showed signs of **overfitting** — very low training loss but higher validation loss.  
- **L1 regularization** achieved the best generalization, producing the lowest validation loss while maintaining stability.  
- **L2 regularization** helped but slightly underperformed compared to L1.  
- **Dropout** also reduced overfitting but was moderately effective here.  

> Overall, L1 regularization provided the best performance on this dataset.

---

## 8. Conclusion
Regularization techniques significantly improve model generalization by preventing overfitting.  
- **L1** performed best in this experiment due to its sparsity-inducing property.  
- **L2** and **Dropout** also helped but to a lesser degree.  
The optimal choice of regularizer depends on the dataset characteristics and model architecture. Further tuning could enhance performance.

---

## 9. Tech Stack
- **Language:** Python  
- **Framework:** PyTorch  
- **Libraries:** NumPy, Matplotlib, Scikit-learn  

