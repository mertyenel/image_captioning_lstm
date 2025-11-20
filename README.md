#  Image Captioning Architecture with Attention Mechanism

This project is a deep learning solution that automatically generates a contextual text description (caption) by analyzing an image. The work is built on the classic **Encoder-Decoder** architecture, which combines visual and text data.

***

##  Architectural Choice: Ancestors of LLMs and Performance

In this project, instead of modern technologies like **Large Language Models (LLMs)**, the **Sequence-to-Sequence (Seq2Seq)** architecture, which is based on their more primitive ancestor **LSTM** (Long Short-Term Memory), was preferred.

Although this basic architecture has a lower accuracy rate compared to current VLMs, it clearly demonstrated the core solution mechanism of the problem by producing **quite satisfactory** and meaningful results with low computational cost.

***

##  Model Architecture and Vectorization

### 1. Visual Encoder
The Encoder's goal is to convert the image into a matrix of semantic features:
* **Base Model:** The pre-trained **ResNet-50** architecture was used (**Transfer Learning**).
* **Dondurma (Freezing):** The main body (`backbone`) of ResNet was kept frozen.
* **Output Vectors:** The visual input was passed through convolutional layers, then the channel size was fixed to **256** using a $1 \times 1$ convolution, and presented to the Decoder as a **$49 \times 256$** feature matrix. This matrix represents 49 spatial regions of the image.

### 2. Text Decoder
**LSTMCell** was used to generate the text sequence.

* **Vektörleştirme (Embedding):** The **`torch.nn.Embedding`** layer was used as the word vectorizer.
    * This layer functions as a **learnable lookup table** that is **learned from scratch** during the image captioning task, without using an external model (Word2Vec, FastText, etc.).
    * Each word is represented by a **256**-dimensional dense vector, which allows the model to learn word-to-word relationships within the task context.
* **Attention Mechanism:** At each prediction step, the LSTM's Hidden State is compared with the **$49 \times 256$** matrix from the Encoder to create a dynamic **Context Vector** ($1 \times 256$).

**LSTM Input:** The final input vector entering the LSTMCell is created by concatenating the **Context Vector** and the previous step's **Word Embedding Vector** (resulting in a **512**-dimensional vector).
