import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.pre import clean_text

# Step 1: Load job descriptions CSV
df = pd.read_csv("job_descriptions.csv")


# Step 2: Clean the text fields
df["combined"] = df["job_title"] + " " + df["required_skills"] + " " + df["description"]
df["combined"] = df["combined"].apply(clean_text)

# Step 3: Vectorize the cleaned text
vectorizer = TfidfVectorizer()
job_vectors = vectorizer.fit_transform(df["combined"])

# Step 4: Save the vectorizer
joblib.dump(vectorizer, "utils/vectorizer.pkl")

# Step 5: Save the vectorized matrix as model (optional)
joblib.dump(job_vectors, "model/matcher_model.pkl")

# Step 6: Save cleaned jobs CSV for display
df.drop(columns=["combined"], inplace=True)
df.to_csv("model/cleaned_jobs.csv", index=False)

print("✅ Model training complete. Files saved in /utils and /model")
