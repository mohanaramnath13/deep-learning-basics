# Lab 6: Baseline RNN for Sequence Modelling  

## Objective  
To understand and implement a **Recurrent Neural Network (RNN)** for sequential data (text), explore preprocessing and embedding methods, train a baseline RNN using **Keras**, and evaluate its performance while analyzing limitations such as overfitting and vanishing gradients.  

## Theory Overview  

### RNNs vs Feedforward NNs  
- Traditional neural networks process inputs independently without memory.  
- RNNs maintain **hidden states**, allowing them to learn sequential dependencies over time.  

### Hidden State Mechanism  
- The hidden state stores contextual information from previous time steps:  
  **hₜ = tanh(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)**  
- Shared weights (Wₕ, Wₓ) across all time steps enable sequence learning.  
- **tanh** activation helps represent both positive and negative context but can cause **vanishing gradients**.  

### Vanishing Gradient Problem  
- During Backpropagation Through Time (BPTT), gradients diminish exponentially, preventing effective long-term learning.  

### Embedding vs One-Hot Encoding  
- **One-hot encoding** treats words as independent and unrelated.  
- **Word embeddings** capture semantic similarity — placing related words closer in vector space.  
- Example: `king - man + woman ≈ queen`.  

### Padding & Truncating Sequences  
- Ensures all sequences have uniform length (100 tokens).  
- Padding adds zeros; truncation removes extra words — may lead to context loss if key words are truncated.  

### Model Components  
- **Embedding Layer:** Converts word indices into dense vectors (dim = 32).  
- **SimpleRNN Layer:** Processes sequences step-by-step, maintaining temporal context.  
- **Dense Layer (Sigmoid):** Outputs probability for binary sentiment classification.  

### Choices of Loss, Optimizer & Regularization  
- **Loss:** Binary Crossentropy — suitable for binary classification.  
- **Optimizer:** Adam — combines adaptive learning and momentum.  
- **Dropout:** Prevents overfitting by randomly disabling neurons during training.  

## Procedure  
- Loaded **IMDb dataset** using `keras.datasets.imdb`.  
- Limited vocabulary to top 10,000 frequent words (`num_words=10000`).  
- Padded sequences to a fixed length of 100 using `pad_sequences`.  
- Built baseline model:  
