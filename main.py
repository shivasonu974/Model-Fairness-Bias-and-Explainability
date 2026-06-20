
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from lime.lime_tabular import LimeTabularExplainer

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
df = pd.read_csv(url, names=columns)

X = df.drop("income", axis=1)
y = df["income"]
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

importance = model.feature_importances_
features = pd.Series(importance, index=X_train.columns)
top_features = features.nlargest(10)
print(top_features)

plt.figure(figsize=(10,5))
top_features.plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.savefig("feature_importance.png")

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['<=50K', '>50K'],
    mode='classification'
)
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba, num_features=10)
print(exp.as_list())

sensitive = df.loc[X_test.index, "sex"]
male_idx = sensitive == "Male"
female_idx = sensitive == "Female"
male_acc = accuracy_score(y_test[male_idx], pred[male_idx])
female_acc = accuracy_score(y_test[female_idx], pred[female_idx])
print("Male Accuracy:", male_acc)
print("Female Accuracy:", female_acc)
print("Accuracy Difference:", abs(male_acc - female_acc))
