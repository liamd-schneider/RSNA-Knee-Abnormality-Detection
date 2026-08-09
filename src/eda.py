"""Quick sanity-check EDA over the small metadata CSVs (train.csv, train_series.csv).

Run after pulling the metadata locally:
    kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/
    kaggle competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p data/
"""
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]


def main():
    train = pd.read_csv("data/train.csv")
    series = pd.read_csv("data/train_series.csv")

    labeled = train.dropna(subset=LABELS, how="all")

    print("train.csv shape:", train.shape)
    print("studies with at least one non-null label:", len(labeled), "/", len(train))
    print()
    print("label prevalence among labeled studies:")
    print(labeled[LABELS].mean(numeric_only=True).round(3))
    print()
    print("total series:", len(series))
    print("avg series per study:", round(series.groupby("StudyInstanceUID").size().mean(), 2))
    print("plane distribution:")
    print(series["Anatomical_Plane"].value_counts())


if __name__ == "__main__":
    main()
