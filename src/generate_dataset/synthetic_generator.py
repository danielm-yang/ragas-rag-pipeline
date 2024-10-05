import os
import json
from pathlib import Path
from pprint import pprint
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader


class SyntheticTestGenerator:
    def __init__(self, openai_api_key, directory):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.directory = directory
        self.documents = []
        self.testset = None


    # Load documents
    def load_documents(self):
        loader = JSONLoader(
            file_path='./src/data/train-v2.0.json',
            jq_schema='.data[].paragraphs[] | {context: .context, qas: .qas}',
            text_content=False
        )

        self.documents = loader.load()

        for document in self.documents:
            document.metadata['filename'] = document.metadata.get('source', 'unknown_source')

        if self.documents:
            print("Documents successfully loaded.")
        else:
            print("Failed to load documents.")


    # Generate synthetic test set from loaded documents
    def generate_synthetic_test(self):
        generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        critic_llm = ChatOpenAI(model="gpt-4")
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )

        self.testset = generator.generate_with_langchain_docs(self.documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

        print("Testset successfully generated.")

    
    # Export results to Panda dataframe
    def visualize_dataframe(self):
        print("Testset exported to Pandas Dataframe.")
        self.testset.to_pandas();
