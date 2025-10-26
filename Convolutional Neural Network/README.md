# Lab 4: CNN Classification  

## Objective  
To implement and train a 1D Convolutional Neural Network (CNN) using PyTorch for a synthetic image classification task, and to study its performance, overfitting tendencies, and the effect of architectural components such as Batch Normalization.  

## Dataset  
- Synthetic dataset with 1000 samples.  
- Each sample represented as a 1D vector of 32 numerical pixel values.  
- Data split: 80% training (800 samples), 20% testing (200 samples).  
- A portion of the training set was used for validation.  

## Model Architecture  
- **Conv1D Layer 1:** 1 input channel → 16 output channels, kernel size = 3, padding = 1  
- **Activation:** ReLU + **Batch Normalization**  
- **Pooling:** MaxPooling1D (kernel size = 2)  
- **Conv1D Layer 2:** 16 input channels → 32 output channels, kernel size = 3, padding = 1  
- **Activation:** ReLU + **Batch Normalization** + MaxPooling1D  
- **Fully Connected Layer:** 256 → 64  
- **Output Layer:** 64 → 10 (for 10 output classes)  
- Designed for **1D data** instead of 2D images, as there was no spatial structure.  

## Training Configuration  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (learning rate = 0.001)  
- **Batch Size:** 32  
- **Epochs:** 20  
- Monitored training and validation loss and accuracy at each epoch.  

## Results  
| Metric | Training | Validation |
|:--------|:-----------:|:------------:|
| Accuracy | > 80% | 32–35% |

- The model achieved high training accuracy but low validation accuracy, indicating **overfitting**.  
- The **confusion matrix** revealed uneven class predictions, showing weak generalization.  
- Since the data was simple and 1D, convolution layers had **limited feature extraction** capability.  

## Observations  
- Batch Normalization improved convergence and training stability.  
- Extending epochs (10 → 20) raised training accuracy but didn’t improve validation accuracy, confirming overfitting.  
- The small dataset and simple 1D structure restricted learning complexity.  

## Visualization Summary  
- **Training Loss:** Steady decrease over epochs.  
- **Validation Accuracy:** Relatively flat trend, confirming limited generalization.  

## Conclusion  
The experiment successfully demonstrated the implementation and training of a **1D CNN** for classification. The model’s strong performance on the training data but weak generalization on unseen data highlights **overfitting** and the importance of dataset size, model complexity, and regularization techniques. The lab reinforced understanding of CNN structure, feature learning, and the effects of training hyperparameters.
