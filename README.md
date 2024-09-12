
# TRUTH-MATRIX 📰

Truth Matrix is a robust AI-ML model for detecting fake news 📉, trained on a comprehensive dataset of global 🌍 and Indian 🇮🇳 news articles. It leverages a combination of XGBoost 📈, logistic regression 🔍, and Naive Bayes 📊, predicting based majority voting. By analyzing text features with a TF-IDF vectorizer 🧩, it provides reliable classification of news articles to help counteract misinformation 🚫📰.


## Team-Codex

Chetan Sharma : Team Lead and ML developer  
Email : chetan.sharma162004@gmail.com

Viraj Singh : ML Developer  
Email : anandviraj30@gmail.com

Devansh Tiwari : GitHub Manager and FrontEnd Developer  
Email : devanshtiwari2610@gmail.com

Pratham Varma : GitHub Manager and Algoman  
Email : prathamvarma178@gmail.com


## Model Deployment

**Checkout our Website** 🌐 [here](https://fakenewdetectionusing-ml-me2evi96sebrcvuqmpbmvp.streamlit.app/)<br>

-Checkout Google Collab for quick analysis:<br>-[Open file on Colab 📂](https://colab.research.google.com/drive/1DLYD47gZS3bvb-T1Lmpb4M_zGWHgK_az?usp=sharing)<br>
-[Click to Open Statistics of our Model 📈](https://colab.research.google.com/drive/1_7Tu8JmxeUVacG1OP6e_-p54-lE9bvIV?usp=sharing)                               
-[Click to view Performance of different implementations 📊](https://colab.research.google.com/drive/1r7xvbge1FC3lmPizH9Q7CdFoFBbgwURH?usp=sharing)<br><br>
-Checkout the ppt for better understanding : [link comes here]

## Dataset 

The dataset contains 24,529 news articles with an average word count of 75
[(Download the dataset from here)](https://drive.google.com/drive/folders/1ZkP59nvC50pb241gLSIgGGf_615TQM0u?usp=sharing)<br>
Word Cloud:<br>
![download](https://github.com/user-attachments/assets/aa9526c9-456e-41f0-be98-7dedde58489f)<br>

### Dataset Pre-Processing
### 1. **Import Libraries**
   You are importing several important libraries at the beginning of the code:
   - `pandas`, `numpy`: For data manipulation and numeric operations.
   - `re`: For regular expressions, used for text cleaning.
   - `nltk`: Used for natural language processing, especially stopwords removal and stemming.
   - `TfidfVectorizer`: Converts text to a vector representation based on term frequency-inverse document frequency (TF-IDF).

### 2. **Load Datasets**
   You are loading two datasets:
   - `news_dataset.csv`: Contains a labeled dataset of real and fake news (`dataset_1`).
   - `train.csv`: Another dataset with text fields (`dataset_2`).
   - For `dataset_1`, you replace the labels `FAKE` with `1` and `REAL` with `0`.
   - For `dataset_2`, you create a new `text` column by concatenating the `author` and `title` columns and dropping unnecessary columns like `id`, `author`, and `title`.

### 3. **Concatenate Datasets**
   - Both datasets are combined into a single DataFrame `dataset` using `pd.concat`. This allows you to work with one dataset, combining the real and fake news data.

### 4. **Handle Missing Data**
   - `isnull().sum()` is used to check for missing data in both `dataset_1` and `dataset_2`.
   - You fill any missing values in the combined `dataset` with a space (`' '`), ensuring that there are no null values before model training.

### 5. **Stemming Function**
   - You define a `stemming()` function that:
     - Removes non-alphabetical characters.
     - Converts the text to lowercase.
     - Splits the text into words.
     - Removes stopwords (common words like "the", "is").
     - Stems words (e.g., "running" becomes "run") using `PorterStemmer`.
     - Rejoins the words into a cleaned text string.

### 6. **Apply Stemming**
   - You apply the `stemming()` function to the `text` column of the `dataset` to preprocess all the text data.

### 7. **Vectorize Text**
   - `TfidfVectorizer` is used to convert the preprocessed text into a numerical format (TF-IDF vectors) for model training.
   - You `fit_transform` the vectorizer on the `X` dataset (the text column), which learns the vocabulary and transforms the text into vectors.

## Approach
1. **Data Loading and Preprocessing**:
   - Two datasets are loaded: one containing news articles with labels (fake or real) and another with additional text data.
   - Unnecessary columns (like IDs, authors, and titles) are removed from the second dataset.
   - Missing values are filled with empty spaces, and labels are converted to binary (FAKE = 1, REAL = 0).
   - A **stemming** function is applied to clean and preprocess the text, removing unwanted characters and stopwords, and reducing words to their root form.
   - Both datasets are merged into one and the text is transformed using the **TF-IDF vectorizer** to convert text into numerical features.

2. **Train-Test Split**:
   - The data is split into training (80%) and testing (20%) sets using **train_test_split**.
   
3. **Model Training**:
   - Three machine learning models are trained on the training set:
     - **Logistic Regression**
     - **XGBoost**
     - **Naive Bayes**
   - Each model is evaluated on the training and testing sets using **accuracy_score** to calculate prediction accuracy.

4. **Manual Prediction**:
   - A function takes a user's inputted article, applies the same preprocessing (stemming and vectorization), and passes it through all three models.
   - The final prediction is based on a **majority vote** among the three models: if two or more predict "fake news," it is classified as fake, otherwise, it is classified as real.

5. **Output**:
   - The program displays the prediction (REAL or FAKE) for the input article and provides the accuracy of the models on the test data.
     
## Workflow

1. Data Preparation
2. Pre Processing
3. Vectorization
4. Splitting the data
5. Training The models
6. Testing with manual Inputs
## Modifications

### Brief analysis

Train Data accuracy:-
1)Logistics regression : 98.68%
2)XgBoost: 99.25%
3)Naive Bayes: 97.78%

Test Data accuracy:-
1)Logistics regression : 97.92%
2)XgBoost: 98.73%
3)Naive Bayes: 96.35%

Custom Test:-
We performed a test consisting of 15 different news articles, and the results for each approach are as follows:
Logistics + XGBoost: 12/15 correct (80% accurate)
Logistics + XGBoost + NB: 14/15 correct (93% accurate)
Logistic Regression: 12/15 correct (80% accurate)

## Results
Let's see a demo on our website with a fake sample article:<br>
In June 2020, Baba Ramdev claimed, “We have prepared the first Ayurvedic-clinically controlled, research, evidence, and trial-based medicine for COVID-19. We conducted a clinical case study and clinical controlled trial, and found that 69% of the patients recovered within three days and 100% recovered within seven days.<br>
![image](https://github.com/user-attachments/assets/8da7555f-c0ed-4d17-82ae-dae3d3382626)<br>

As predicted we can see that the model correctly gave an output that the news was fake.<br>

### Graphs
Confusion matrix:<br>
![download](https://github.com/user-attachments/assets/aba56701-1098-44b7-a2f5-0ad248b20bdd)<br>
![download](https://github.com/user-attachments/assets/a849645b-13f5-4b54-8de0-e4c2a4cd5169)<br>
![download](https://github.com/user-attachments/assets/e2085bf1-fb6a-4dc1-9ee3-b07da7c4e63e)<br>
Model Accuracy:<br>
![download](https://github.com/user-attachments/assets/9d543059-6459-47b4-a354-26b047428e8b)<br>
Logistic Regression Learning Curve:<br>
![download](https://github.com/user-attachments/assets/14c3954c-5061-41fc-890f-339bc953a8f0)<br>
Feature Importances of XGBOOST:<br>
![image](https://github.com/user-attachments/assets/8a44765d-4a63-40c7-92b2-0600ddae0174)<br>

## Video Explanation 

(Drag and drop the video file where you detect fake news)
## Disclaimer

This model is trained and tested for recognizing certain words which may indicate that the news is fake and doesn't track the context and the current affairs.
