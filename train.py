from datasets import load_dataset

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
