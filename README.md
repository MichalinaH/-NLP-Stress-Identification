# Stress Identification

Repository contains 8 files:
1. .gitignore 
2. 1_StressIdentification_rekurencyjna.ipynb <--- notebook with data cleaning and class distribution, LSTM model and its usage 
3. 2_StressIdentification_CNN.ipynb <--- notebook with data cleaning and class distribution, simple CNN model and its usage 
4. 3_StressIdentification_pre-train.ipynb <--- notebook with data cleaning and class distribution, pre-train word2vec model and its usage 
5. 4_StressIdentification_fine_tune.ipynb <--- notebook with data cleaning ,[DistilBERT base model](https://huggingface.co/distilbert-base-uncased) and its usage 
6. README.md 
7. Stress.csv <--- dataset
8. requirements.txt <--- file with everything You need to have to run this notebooks

**Dataset:** https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction

The dataset contains data posted on subreddits related to mental health. This dataset contains various mental health problems shared by people about their life. 
Fortunately, this dataset is labelled as 0 and 1, where **0** indicates **no stress** and **1** indicates **stress**.

To properly use this document, you must install requirements.txt file: 
```shell
pip install -r requirements.txt
```

## Description of the Chosen Models
* [LSTM RNN](../-NLP-Stress-Identification/1_StressIdentification_rekurencyjna.ipynb):

    The model architecture consists of:

    **Embedding Layer**: Utilizes pre-trained GloVe word embeddings of 100 dimensions to convert words into fixed-size dense vectors, aiding in understanding word meanings and contexts.

    **LSTM Layers**: Employs Long Short-Term Memory (LSTM) units, a type of recurrent neural network, to capture long-range dependencies within text sequences. The model comprises two LSTM layers, the first with 128 units and the second with 64 units, allowing for the understanding of sequential patterns in text data.

    **Dense Layers**: Follows the LSTM units and includes two dense layers. The first layer with 32 units uses the rectified linear unit (ReLU) activation function, while the final layer with a single unit employs the sigmoid activation function for binary classification.

    **Training Setup**: The model is compiled using the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric. Training occurs over 9 epochs with a learning rate reduction strategy based on validation accuracy, ensuring effective learning and generalization.

<br>

* [Simple convolutional neural network (CNN) with following layers](../-NLP-Stress-Identification/2_StressIdentification_CNN.ipynb):
  
    The model architecture consists of:
    **Embedding Layer**: Incorporates an Embedding layer initialized with a 50-dimensional representation, efficiently converting words into fixed-size dense vectors to capture word meanings and context in the input text data.

    **Convolutional Layers (Conv1D)**: Utilizes a one-dimensional convolutional layer with 128 filters and a kernel size of 5, employing the rectified linear unit (ReLU) activation function to detect essential patterns and local relationships within the text sequences.

    **Global Max Pooling Layer**: Implements a GlobalMaxPooling1D layer to extract the most significant features from the convolutional layer's output, aggregating and preserving essential information throughout the text sequences.

    **Dense Layer**: Includes a single dense layer with a sigmoid activation function, aiding in binary classification by producing an output between 0 and 1, predicting the presence or absence of stress in a given text.

    **Training Setup**: The model is compiled using the Adam optimizer and the binary cross-entropy loss function. Throughout the training process, the accuracy metric is tracked. The model is trained over 3 epochs to learn and generalize effectively from the provided data.

<br>

* [Pre-Trained Word Embeddings with word2vec model](../-NLP-Stress-Identification/3_StressIdentification_pre-train.ipynb):
  
    This model incorporates pre-trained Word2Vec embeddings obtained from the Google News dataset, enabling the representation of words in a continuous vector space of 300 dimensions. The Word2Vec embeddings capture semantic relationships between words based on their contextual usage within a vast corpus.

    **Model Architecture**:
  
    **Embedding Layer**: Utilizes the pre-trained Word2Vec embeddings as fixed weights, ensuring that the model benefits from the rich semantic information present in the embedding vectors. The layer transforms input text sequences into embedded representations.

    **LSTM Layer**: Employs a Long Short-Term Memory (LSTM) neural network with 128 units. LSTMs are adept at capturing long-range dependencies within sequences, enabling the model to comprehend the contextual nuances present in the text.

    **Dense Layer (Output)**: Consists of a single unit with a sigmoid activation function, facilitating binary classification. This layer predicts whether the input text signifies stress or not.

    **Training Setup**: The model is compiled using the Adam optimizer and binary cross-entropy loss function. It undergoes training over 6 epochs with a batch size of 32.

<br>

* [Language Model Fine-Tuning with DistilBERT base model](../-NLP-Stress-Identification/4_StressIdentification_fine_tune.ipynb):
  
    **Pre-trained DistilBERT**: Utilizes the pre-trained DistilBERT model, which serves as the backbone for this sentiment analysis task. DistilBERT is fine-tuned for sequence classification, specifically for sentiment analysis.
    <br>

    DistilBERT:
    A compact version of BERT, designed by Hugging Face.
    Retains BERT's understanding of context from pre-training.
    Smaller, faster, and resource-efficient.
    Ideal for sentiment analysis due to its contextual comprehension and efficient performance, achieved through fine-tuning on sentiment-labeled data.


<br>


| Model                        | Accuracy      | Did model predict correctly on test function  |
| -----------------------------|:-------------:| ---------------------------------------------:|
| LSTM RNN                     | 65.60%        | Yes                                           |
| Simple CNN                   | 70.92%        | Yes                                           |
| Word2vec model               | 69.50%        | Yes                                           |
| Pre-trained DistilBERT model | 84.85%        | Yes                                           |

## Model usage
1. Install Required Libraries
2. Chose notebook you would like to test 
3. Run the Entire File by Clicking the `Run All` Button
4. In the Last Cell, Locate the `sentence` Variable. Replace the Placeholder Sentence with Your Own Text that You Want to Classify
 ```shell
sentence = "I had a peaceful evening reading my favorite book."
```
