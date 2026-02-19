from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

ds = load_dataset("Tobi-Bueck/customer-support-tickets")

df = ds["train"].to_pandas()
print(df.shape)

df = df[df["language"] == "en"].copy()
print(df["type"].value_counts())

df = df[["subject", "body", "type"]]
print(df.shape)

# delete the rows without text
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["text"] = df["subject"] + " " + df["body"]
df["text"] = df["text"].str.strip()
df = df[df["text"] != ""]
print("empty texts:", (df["text"].str.strip() == "").sum())
print(df.shape)

print("rows:", df.shape[0])

print(df[["type", "text"]].sample(2, random_state=0).to_string(index=False))

X = df["text"]
y = df["type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("train size :", X_train.shape, "test size :", X_test.shape)

model = Pipeline([("tifidf", TfidfVectorizer(stop_words="english")), ("clf", LogisticRegression(max_iter=1000))])

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("f1 macro :", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_ )
plt.xlabel("predicted")
plt.ylabel("true")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")

print("classes:", model.classes_)
print("proba example:", model.predict_proba([df["text"].iloc[0]])[0])