import os
from datasets import load_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate


class SyntheticTestEvaluator:
    def __init__(self, openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.dataset = []
        self.result = None


    # Load dataset
    def load_in_dataset(self):
        self.dataset = load_dataset("explodinggradients/amnesty_qa", "english_v2")
        
        if self.dataset:
            print("Dataset successfully loaded.")
        else:
            print("Failed to load dataset. Aborting...")
            exit(1)


    # Evaluate metrics on dataset
    def evaluate_metrics(self):
        self.result = evaluate(
            self.dataset["eval"],
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )

        print("Dataset successfully evaluated.")


    def visualize_dataframe(self):
        print("Results exported to Pandas Dataframe.")
        df = self.result.to_pandas()
        df.head()
