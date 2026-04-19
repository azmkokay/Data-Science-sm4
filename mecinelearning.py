from sklearn.tree import DecisionTreeClassifier

# Data training
# [jam belajar, kehadiran]
X = [
    [2, 60],
    [3, 70],
    [4, 75],
    [5, 80],
    [6, 90],
    [1, 50]
]

# Target (0 = tidak lulus, 1 = lulus)
y = [0, 0, 1, 1, 1, 0]

# Model Machine Learning
model = DecisionTreeClassifier()

# Training
model.fit(X, y)

# Prediksi siswa baru
prediksi = model.predict([[4, 80]])

print("Prediksi Lulus:", prediksi)