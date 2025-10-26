# Comparison of RNN, LSTM, and GRU on Text Classification

## 1. Objectives
- Understand architectural differences between RNN, LSTM, and GRU.
- Implement models using Keras for text classification.
- Compare performance in terms of training speed, accuracy, and limitations.

## 2. Dataset Preparation
- Text data was cleaned, tokenized, and padded to a uniform sequence length.
- The dataset was split into training and testing sets with balanced class representation.

## 3. Model Architecture
Each model had the following structure:
Embedding → Recurrent Layer (RNN/LSTM/GRU) → Dense(1, activation='sigmoid')

- Model 1: SimpleRNN
- Model 2: LSTM
- Model 3: GRU
- All models trained for 5 epochs using identical hyperparameters.

## 4. Evaluation Results
| Model       | Test Accuracy |
|--------------|---------------|
| SimpleRNN    | 0.7910        |
| LSTM         | 0.9302        |
| GRU          | 0.9319        |

## 5. Discussion
- SimpleRNN: Fastest training but suffers from vanishing gradients, limiting long-term learning.
- LSTM: Captures long-term dependencies effectively using input, forget, and output gates. Highest accuracy but slowest training.
- GRU: Combines efficiency and performance with fewer parameters. Nearly matches LSTM accuracy while training faster.

## 6. When to Use
- SimpleRNN: Lightweight tasks or limited compute resources.
- LSTM: Complex sequential tasks demanding high accuracy.
- GRU: Balanced choice for performance and efficiency on larger datasets.

## 7. Summary
This experiment demonstrates that architectural innovations in recurrent units directly affect model capability.
While SimpleRNN provides a baseline, LSTMs and GRUs outperform it significantly in text classification tasks.
GRU offers the best trade-off between computational cost and accuracy, making it ideal for scalable NLP applications.
