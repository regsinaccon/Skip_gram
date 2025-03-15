import numpy as np
from numba.experimental import jitclass
from numba import int64 ,float64
from collections import defaultdict
import string
from matplotlib.pyplot import *
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt') 
nltk.download('stopwords') 
nltk.download('punkt_tab')



def add_coustom_stop_word(*args,stop_words):
  for word in args:
    stop_words.append(word)
    return stop_words



def tokenize_text(file_path):
  """
  tokenize words in the text to numbers
  """
  with open(file_path,'r') as txt:
    text = txt.read()
    text.replace('\n',' ')
    text.lower()
  text.translate(str.maketrans('','',string.punctuation))
  word_tokens = word_tokenize(text)
  tokens_after_filtering = []

  stop_words = add_coustom_stop_word('the','.',stop_words=stopwords.words('english'))
  stop_words = set(stopwords.words('english'))

  for token in word_tokens:

    if token not in stop_words and token.isnumeric() == False:
      tokens_after_filtering.append(token.lower())


  return list(set(tokens_after_filtering))


def get_index(word,tokens)->int:
  return tokens.index(word)

def get_word(index,tokens):
  return tokens[int(index)]



def text2token(file_path,text2token_table):
  indice = np.array([],dtype=int)
  with open (file_path,'r') as txt:
    text = txt.read()
  for word in text.split(' '):
    if word in text2token_table:
      indice = np.append(indice,get_index(word ,text2token_table))
  return indice

tokens2word_table = tokenize_text('testdata2.txt') #Create a word and number translating table () data type == list using its index to determine the numeric form of the word
indice = text2token('testdata2.txt',tokens2word_table) #Turn each word in the text to number
# print(tokens2word_table[0])


# for i in range(len(text2token('testdata.txt',tokenize_text('testdata.txt')))):
#   print(text2token('testdata.txt',tokens2word_table)[i],get_word(text2token('testdata.txt',tokens2word_table)[i],tokens2word_table))




spec = [
    ('vocab_size', int64),
    ('embedding_dim', int64),
    ('learning_rate', float64),
    ('batch_size', int64),
    ('total_loss',float64),
    ('W1',float64[:,:]),
    ("W2",float64[:,:]),
    ("dW1",float64[:,:]),
    ("dW2",float64[:,:])
]

@jitclass(spec)
class SkipGram():
    def __init__(self, vocab_size:int, embedding_dim:int, learning_rate=0.03,batch_size=45):
        """
        Initialize Skip-gram model parameters

        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of word embeddings
            learning_rate (float): Learning rate for gradient descent
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # W1: embedding matrix (vocab_size × embedding_dim)
        # W2: output matrix (embedding_dim × vocab_size)
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)
        self.total_loss = 0

    def softmax(self, x):
        """Compute softmax values for each set of scores"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def generate_training_data(self, corpus, window_size):
        """
        Generate training pairs (target word, context word)

        Args:
            corpus (list): List of word indices
            window_size (int): Size of context window
        """
        training_data = []

        for i, target_word in enumerate(corpus):
            # Define context window boundaries
            start = max(0, i - window_size)
            end = min(len(corpus), i + window_size + 1)

            # Generate (target, context) pairs
            for j in range(start, end):
                if j != i:
                    context_word = corpus[j]
                    training_data.append((target_word, context_word))

        return training_data

    def forward(self, target_word_idx):
        """
        Forward pass

        Args:
            target_word_idx (int): Index of target word
        """
        # Get target word embedding
        h = self.W1[target_word_idx]  # (embedding_dim,)

        # Compute scores
        u = np.dot(self.W2.T, h)  # (vocab_size,)

        # Apply softmax
        y_pred = self.softmax(u)  # (vocab_size,)

        return h, u, y_pred

    def backward(self, target_word_idx, context_word_idx, h, y_pred):
        """
        Backward pass and weight updates

        Args:
            target_word_idx (int): Index of target word
            context_word_idx (int): Index of context word
            h (np.array): Hidden layer output
            y_pred (np.array): Predicted probabilities
        """
        # Compute error
        e = y_pred.copy()
        e[context_word_idx] -= 1  # (vocab_size,)

        # Compute gradients
        dW2 = np.outer(h, e)  # (embedding_dim × vocab_size)
        dW1 = np.dot(self.W2, e)  # (embedding_dim,)

      # Add gradient 
        self.dW1 += dW1  
        self.dW2 += dW2
        np.clip(self.dW1,-1,1,out=self.dW1)
        np.clip(self.dW2,-1,1,out=self.dW2)


    def train(self, corpus, window_size, epochs):
        """
        Train the Skip-gram model

        Args:
            corpus (list): List of word indices
            window_size (int): Size of context window
            epochs (int): Number of training epochs
        """
        training_data = self.generate_training_data(corpus, window_size)
        
        
        for epoch in range(epochs):

            for i in range(self.batch_size):
              self.dW1 = np.zeros_like(self.W1)
              self.dW2 = np.zeros_like(self.W2)
              for target_word, context_word in training_data:
                  # Forward pass
                  h, u, y_pred = self.forward(target_word)

                  # Compute loss (negative log likelihood)
                  self.total_loss = -np.log(y_pred[context_word])


                  
                  self.backward(target_word, context_word, h, y_pred)
              # update weight
              self.W1 -= self.learning_rate/self.batch_size * self.dW1
              self.W2 -= self.learning_rate/self.batch_size * self.dW2

            print("Epoch", epoch + 1, "Loss:",self.total_loss)



    def get_embedding(self, word_idx):
        """Get embedding for a specific word
        Args:
            word_idx (int): Index of the word
        You can use get_embedding after load_weitht to vectorize word
        """
        return self.W1[word_idx]

    def load_weight(self,W1_path=None,W2_path=None):
        """
        Load pre-trained weights with file name "Weight1.npy" and "Weight2.npy" as defult
        You can use forward after load_weight to vectorize word
        """
        if W1_path == None:
            W1_path = "Weight1.npy"
        if W2_path == None:
            W2_path = "Weight2.npy"
        self.W1 = np.load(W1_path)
        self.W2 = np.load(W2_path)


    



if __name__ == "__main__":

    corpus = indice  
    vocab_size = len(tokens2word_table)  # Number of unique words
    embedding_dim = 40
    window_size = 3
    epochs = 30
    learning_rate = 0.03
    # Initialize and train model
    model = SkipGram(vocab_size, embedding_dim,batch_size=45,learning_rate=learning_rate)
    model.train(corpus, window_size, epochs)
    #Save weight in binary form
    np.save("Weight1.npy",model.W1) 
    np.save("Weight2.npy",model.W2)
