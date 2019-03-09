from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [1., 1.]]
y = [0, 1]

clf = MLPClassifier(solver='sgd', alpha=1e-5, activation='logistic',
                    hidden_layer_sizes=(5, 2), max_iter=2000, tol=1e-4)
clf.fit(X, y)

predicted_value = clf.predict([[2., 2.], [-1., -2.]])
print(predicted_value)
predicted_proba = clf.predict_proba([[2., 2.], [-1., -2.]])
print(predicted_proba)

print([coef.shape for coef in clf.coefs_])
print([coef for coef in clf.coefs_])
