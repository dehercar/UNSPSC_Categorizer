import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer

def run_unspsc(df2,desc_clean,predicted_col_name):
    #delete this line for power bi
    start = timer()
    np.random.seed(500)
    #delete this line for power bi
    #delete this line for power bi
    df = pd.read_csv("Family_Training_Set.csv")
    #delete this line for power bi

    df = shuffle(df,random_state=0)

    x = df['Description']
    y = df['UNSPSC']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    vectorizer = TfidfVectorizer(ngram_range=(1,3))
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    test_vectors_2 = vectorizer.transform(df2[desc_clean])
    clf = LinearSVC()
    clf.fit(train_vectors,y_train)
    predicted = clf.predict(test_vectors)
    acc = accuracy_score(y_test,predicted)
    #delete this line for power bi
    print('Model Acccuracy SVC')
    #delete this line for power bi
    print(acc*100)
    results = clf.predict(test_vectors_2)
    df2[predicted_col_name] = results
    df2[predicted_col_name] = df2[predicted_col_name].astype(str)
    #delete this line for power bi
    end = timer()
    #delete this line for power bi
    print(end - start)
    return df2
