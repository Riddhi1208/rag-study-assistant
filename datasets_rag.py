from datasets import load_dataset

ds = load_dataset(
    "google-research-datasets/natural_questions",
    "default",
    split="train",
    streaming=True
)

for i, item in enumerate(ds.take(10)):
    print(f"\nQuestion {i + 1}:")
    print(item["question"])