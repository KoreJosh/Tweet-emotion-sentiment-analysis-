![](tweet_emotion.ipynb)

# Tweet-emotion-sentiment-analysis-


This repository contains the code and analysis for preprocessing and analyzing tweet emotion data. The dataset used in this project was obtained from Kaggle and contains tweets labeled with various emotions such as happiness, sadness, anger, and more. The goal of this project is to clean, preprocess, and analyze the data to understand the distribution of emotions and prepare it for further machine learning tasks.
---

Table of Contents

Overview

Dataset

Requirements

Project Workflow

1. Data Loading

2. Data Preprocessing

3. Exploratory Data Analysis (EDA)

---

Overview

The goal of this project is to preprocess and analyze tweet data to classify emotions such as happiness, sadness, anger, and more. This helps in understanding public sentiment or emotion trends in social media posts.

---

Dataset

The dataset for this project was sourced from Kaggle and contains thousands of labeled tweets indicating specific emotions.


---

Requirements

Install the required libraries using the command below:

pip install -r requirements.txt

Some of the main libraries used in this project include:

pandas

numpy

matplotlib

seaborn

nltk



---

Project Workflow

1. Data Loading

The dataset is loaded using pandas. Here's an example snippet:
# Dataset

The dataset used in this project is the **Tweet Emotion Dataset** from Kaggle. It contains 40,000 tweets labeled with 13 different emotions:

- **neutral**
- **worry**
- **happiness**
- **sadness**
- **love**
- **surprise**
- **fun**
- **relief**
- **hate**
- **empty**
- **enthusiasm**
- **boredom**
- **anger**

The dataset is stored in a CSV file with the following columns:

- `tweet_id`: Unique identifier for each tweet.
- `sentiment`: The emotion label for the tweet.
- `content`: The text content of the tweet.


import pandas as pd

# Load the dataset
data = pd.read_csv("tweets_emotion.csv")
print(data.head())

This displays the first few rows of the dataset to understand its structure.


---

2. Data Preprocessing

Preprocessing involves cleaning the tweets, removing unnecessary elements like URLs, punctuation, and converting text to lowercase.

import re
from nltk.corpus import stopwords

# Preprocessing Steps

The preprocessing steps are crucial for preparing the text data for analysis and modeling. Below are the steps taken to clean and preprocess the tweet data:

### 1. **Install Required Libraries**
Before starting, ensure you have the necessary libraries installed. You can install them using pip:

```python
!pip install contractions
```

### 2. **Import Libraries**
We import the necessary libraries for data manipulation, text processing, and visualization.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import seaborn as sns
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 3. **Load the Dataset**
Load the dataset from the CSV file.

```python
df = pd.read_csv(r'C:\\Users\\user\\Documents\\WorkSpace\\data\\nlp data\\tweet_emotions.csv')
```

### 4. **Exploratory Data Analysis (EDA)**
Before preprocessing, we perform some basic EDA to understand the dataset.

```python
# Display the first few rows of the dataset
df.head()

# Check the distribution of emotions
df.sentiment.value_counts()
```

### 5. **Data Cleaning**
We clean the text data by removing unnecessary elements such as capitalizations, numbers, punctuation, and stopwords.

#### a. **Remove Capitalizations**
Convert all text to lowercase to ensure uniformity.

```python
def removeCaps(text):
    text = text.lower()
    return text

df['contRCaps'] = df.content.apply(removeCaps)
```

#### b. **Remove Numbers**
Remove any digits from the text.

```python
def removeNums(text):
    return ''.join([char for char in text if not char.isdigit()])

df['textNDigits'] = df["contRCaps"].apply(lambda text: removeNums(text))
```

#### c. **Remove Punctuation**
Remove punctuation marks from the text.

```python
df['contNPunc'] = datastriper(df.textNDigits, string.punctuation)
```

#### d. **Remove Short Words**
Remove words with a length of 2 or fewer characters.

```python
bow = []
for row in df.contNPunc:
    bow.extend(row.split(' '))
shrtWrds = [x for x in bow if len(x)<=2]
shrtWords = set(shrtWrds)

df['contNSym'] = datastriper(df.contNPunc, shrtWords)
```

#### e. **Remove Usernames**
Remove Twitter usernames (words starting with '@').

```python
userN = []
for row in df.contNSym:
    wrds = row.split()
    for x in wrds:
        if x.startswith('@'):
            userN.append(x)
```

### 6. **Lemmatization and Stemming**
Lemmatization and stemming are used to reduce words to their base or root form.

```python
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Example of lemmatization
df['lemmatized_text'] = df['contNSym'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Example of stemming
df['stemmed_text'] = df['contNSym'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
```

### 7. **Visualization**
Visualize the distribution of emotions using a bar plot.

```python
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df, order=df['sentiment'].value_counts().index)
plt.title('Distribution of Emotions in Tweets')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

### 8. **Word Cloud**
Generate a word cloud to visualize the most common words in the dataset.

```python
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['contNSym']))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Tweets')
plt.show()
```

## Next Steps

After preprocessing, the data is ready for further analysis or modeling. Potential next steps include:

- **Feature Extraction**: Use TF-IDF or word embeddings to convert text into numerical features.
- **Model Training**: Train a machine learning model to classify tweets based on emotion.
- **Evaluation**: Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

## Conclusion

This project demonstrates the steps required to preprocess and analyze tweet emotion data. By cleaning and transforming the text data, we can prepare it for more advanced machine learning tasks. The code provided in this repository can be used as a starting point for similar text analysis projects.

## Acknowledgments

- **Kaggle** for providing the dataset.
- **NLTK** for providing the necessary tools for text processing.
- **Seaborn** and **Matplotlib** for data visualization.
- 
---

Feel free to contribute to this project by opening an issue or submitting a pull request. Happy coding!

