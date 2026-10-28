from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path(__file__).parent / "data" / "metinler.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["metin"],
        df["kategori"],
        test_size=0.25,
        random_state=42,
        stratify=df["kategori"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Veri kümesi: {len(df)} metin")
    print(f"Eğitim örneği: {len(X_train)} | Test örneği: {len(X_test)}")
    print(f"Doğruluk (accuracy): {accuracy_score(y_test, predictions):.2%}")
    print("\nSınıflandırma raporu:")
    print(classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()
