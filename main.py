import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


# Train data reading
print('Train data reading...')
df = pd.read_csv('E:/Datasets/ods_receipts_recognition_Alfabank/train_supervised_dataset.csv')
df = df.fillna(value='unknown')

# Data tokenization
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['name'])

# TFIDF transformation
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Target preparation
LY1 = list(set(df['good']))
LY2 = list(set(df['brand']))
Y1 = [LY1.index(df['good'].iloc[i]) for i in range(len(df))]
Y2 = [LY2.index(df['brand'].iloc[i]) for i in range(len(df))]

# Modelling
print('Model training...')
model1 = OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=12).fit(X_train_tfidf, Y1)
model2 = OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=12).fit(X_train_tfidf, Y2)
print('Model trained')

# Test dataset reading
print('Test data reading...')
df_test = pd.read_csv('E:/Datasets/ods_receipts_recognition_Alfabank/test_dataset.csv')

# Data tokenization
X_test_counts = count_vect.transform(df_test['name'])

# TFIDF transformation
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Prediction
print('Data prediction...')
predicted1 = model1.predict(X_test_tfidf)
predicted2 = model2.predict(X_test_tfidf)
print('Prediction done')

# Result preparation
Out1 = [LY1[predicted1[i]] for i in range(len(df_test))]
Out2 = [LY2[predicted2[i]] for i in range(len(df_test))]

# Сохраняем результат
df_out = pd.DataFrame(list(zip(df_test['id'], Out1, Out2)),
                     columns=['id', 'good', 'brand'])
df_out.to_csv('E:/Datasets/ods_receipts_recognition_Alfabank/result.csv', index=False)
print('Data saved')