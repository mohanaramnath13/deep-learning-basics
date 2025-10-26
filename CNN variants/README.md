# Lab 5: CNN Architecture   

## Objective  
To implement and evaluate multiple **Convolutional Neural Network (CNN)** architectures by varying filters, kernel sizes, and pooling strategies, and to observe how these architectural choices affect model performance and training behavior.  

## Procedure  
- Loaded an image classification dataset (e.g., **MNIST** or **CIFAR-10**) using `torchvision.datasets`.  
- Applied normalization and transformations using `transforms.Compose`.  
- Constructed a **baseline CNN** with two convolutional layers, ReLU activations, MaxPooling, and a fully connected output layer.  
- Split the dataset into **training** and **testing** sets.  
- Created several CNN variants by modifying:  
  - Number of convolutional layers  
  - Filter counts (16 → 64)  
  - Kernel sizes (3×3 vs. 5×5)  
  - Pooling methods (MaxPooling vs. AvgPooling)  
- Trained and evaluated each variant independently, recording accuracy and loss for comparison.  

## Training Configuration  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Metrics Tracked:** Training and validation accuracy, training and validation loss  
- Each model was trained for the same number of epochs to ensure fair comparison.  

## Results and Observations  
| CNN Variant | Filters | Kernel | Pooling | Validation Accuracy | Remarks |
|:-------------|:--------:|:--------:|:----------:|:------------------:|:-----------|
| Baseline | 16 | 3×3 | MaxPooling | 85% | Simple and stable |
| Variant 1 | 64 | 3×3 | MaxPooling | **88%** | Best performing |
| Variant 2 | 64 | 5×5 | MaxPooling | 86% | Lost fine details |
| Variant 3 | 64 | 3×3 | AvgPooling | 84% | Slightly lower accuracy |

- Adding a **third convolutional layer** improved accuracy but increased training time.  
- Increasing filters from 16 to 64 enhanced feature extraction and boosted validation accuracy.  
- **3×3 kernels** captured fine-grained spatial patterns better than 5×5 kernels.  
- **MaxPooling** consistently outperformed AvgPooling in accuracy.  
- The best trade-off was achieved with **3 conv layers, 64 filters, 3×3 kernels, and MaxPooling**, reaching ~88% validation accuracy.  

## Visualization  
- **Validation Accuracy Plot:** Showed consistent improvement across more complex architectures.  
- **Validation Loss Plot:** Decreased steadily for models with more filters and optimized configurations.  

## Conclusion  
This lab highlighted how CNN **architectural choices** — such as filter count, kernel size, and pooling strategy — directly influence model accuracy, generalization, and training time.  
The **optimal architecture** balanced complexity and efficiency, achieving 88% validation accuracy with 3×3 kernels and MaxPooling.  
Through this experiment, a deeper understanding was developed of how architectural tuning can enhance CNN performance and generalization.  
