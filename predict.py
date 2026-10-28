from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path(__file__).parent / "data" / "metinler.csv"

def build_model() -> Pipeline:
    df = pd.read_csv(DATA_PATH)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(df["metin"], df["kategori"])
    return model

def main():
    model = build_model()
    print("Türkçe Metin Sınıflandırma Sistemi")
    print("Bir metin gir; çıkmak için q yaz.\n")
    while True:
        text = input("Metin: ").strip()
        if text.lower() == "q":
            print("Görüşürüz!")
            break
        if not text:
            print("Lütfen boş olmayan bir metin gir.")
            continue
        prediction = model.predict([text])[0]
        print(f"Tahmin edilen kategori: {prediction}\n")

if __name__ == "__main__":
    main()
