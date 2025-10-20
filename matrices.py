import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier  
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('cumulative.csv')

X = data.drop(columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_tce_delivname'])

y = data['koi_disposition']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

#vers on vscode
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#visualisation of cm
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels\n\n' + '\n'.join(y.unique()))
plt.ylabel('True labels\n\n' + '\n'.join(y.unique()))

plt.title('Confusion Matrix - Exoplanet Detection')
plt.show()