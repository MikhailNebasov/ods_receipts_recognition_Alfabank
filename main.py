import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


# Train data reading
print('Train data reading...')
df = pd.read_csv('E:/Datasets/ods_receipts_recognition_Alfabank/train_supervised_dataset.csv')

# Data tokenization
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['name'])

# TFIDF transformation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

Y_train_counts1 = count_vect.fit_transform(df['good'])
Y_train_counts2 = count_vect.fit_transform(df['brand'])

# Modelling
print('Model training...')
model = OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=12).fit(X_train_tfidf, Y_train_counts1)
print('Model trained')

dataset = df['text_fields']
shop_list = df['shop_title']