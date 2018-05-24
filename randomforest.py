import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


print("load data")
x_train = pd.read_csv('./data/train_data.csv')
y_train = pd.read_csv('./data/train_label.csv');
x_test = pd.read_csv('./data/test_data.csv');
y_test = pd.read_csv('./data/test_label.csv');

x_train.drop(x_train.columns[[0, 4, 7]], axis=1, inplace=True)
x_test.drop(x_test.columns[[0, 4, 7]], axis=1, inplace=True)


print("Train Random Forest")
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(x_train, y_train)

print("Predict the result")
predict = rf.predict(x_test)
accuracy = accuracy_score(y_test, predict)

print('Out-of-bag score estimate: {rf.oob_score_:.3}')
print('Mean accuracy score: {accuracy:.3}')

cm = pd.DataFrame(confusion_matrix(y_test, predict), columns=[0, 1], index=[0, 1])
sns.heatmap(cm, annot=True)
