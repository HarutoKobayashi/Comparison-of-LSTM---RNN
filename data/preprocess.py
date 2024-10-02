import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def text_cleaner(text: str) -> str:
    text_clean = re.sub(r"[\"\'.,:;\(\)#\|\*\+\!\?#$%&/\]\[\{\}]", "", text)
    text_clean = re.sub("[0-9]+", "0", text_clean)
    text_clean = re.sub("\s-\s", " ", text_clean)
    return text_clean


col_names = (
    "ID",
    "TITLE",
    "URL",
    "PUBLISHER",
    "CATEGORY",
    "STORY",
    "HOSTNAME",
    "TIMESTAMP",
)
target = (
    "Reuters",
    "Huffington Post",
    "Businessweek",
    "Contactmusic.com",
    "Daily Mail",
)

if __name__ == "__main__":
    with open(DATA_DIR + "/newsCorpora.csv") as original_file:
        df = pd.read_csv(original_file, sep="\t", names=col_names)

    # Use only a portion of the data
    target_df = df[df["PUBLISHER"].isin(target)]
    target_df = target_df.sample(frac=1)

    # Cleaning dataset
    target_df["TITLE"] = target_df["TITLE"].apply(text_cleaner)

    # Split data into train/valid/test
    train_data, tmp = train_test_split(
        target_df, train_size=0.8, stratify=target_df["CATEGORY"]
    )
    valid_data, test_data = train_test_split(
        tmp, train_size=0.5, stratify=tmp["CATEGORY"]
    )

    # Save splitted dataset
    with open(DATA_DIR + "/train.txt", "w") as f:
        train_data[["CATEGORY", "TITLE"]].to_csv(f, header=None, index=None, sep="\t")

    with open(DATA_DIR + "/valid.txt", "w") as f:
        valid_data[["CATEGORY", "TITLE"]].to_csv(f, header=None, index=None, sep="\t")

    with open(DATA_DIR + "/test.txt", "w") as f:
        test_data[["CATEGORY", "TITLE"]].to_csv(f, header=None, index=None, sep="\t")
