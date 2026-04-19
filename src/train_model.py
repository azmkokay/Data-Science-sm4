import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

print("Loading features...")
train = pd.read_csv("dataset/train_features.csv")
test = pd.read_csv("dataset/test_features.csv")

X_train = train.iloc[:, :-1].values
y_train = train['label'].values
X_test = test.iloc[:, :-1].values
y_test = test['label'].values

model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Evaluasi
pred = model.predict(X_test)
print(f"✅ Accuracy: {accuracy_score(y_test, pred)*100:.2f}%")

joblib.dump(model, "models/face_shape_classifier.pkl")
print("✅ Model berhasil disimpan di models/face_shape_classifier.pkl")