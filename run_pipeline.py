import argparse
from src.generate_dataset import SyntheticTestGenerator
from src.evaluate_dataset import SyntheticTestEvaluator


def run_generator(openai_api_key):
    directory = "your_directory"

    generator = SyntheticTestGenerator(openai_api_key)

    generator.load_documents()

    generator.generate_synthetic_test()

    generator.visualize_dataframe()


def run_evaluator(openai_api_key, testset):
    directory = "your_directory"

    evaluator = SyntheticTestEvaluator(openai_api_key, testset)

    evaluator.load_in_dataset()

    evaluator.evaluate_metrics()

    evaluator.visualize_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Run parts of the RAG pipeline.")
    parser.add_argument('--run', choices=['generator', 'evaluator', 'all'], default='all', help="Which part of the RAG pipeline to run?")

    args = parser.parse_args()

    openai_api_key = input("Enter your OpenAI API key: ")

    if args.run == 'generator':
        print("Running synthetic test generator...")
        run_generator(openai_api_key)

    if args.run == 'evaluator':
        print("Evaluating test set...")
        run_evaluator(openai_api_key)

    if args.run == 'all':
        print("Running synthetic test generator...")
        testset = run_generator(openai_api_key)

        print("Evaluating test set...")
        run_evaluator(openai_api_key=openai_api_key, testset=testset)


if __name__ == '__main__':
    main()
