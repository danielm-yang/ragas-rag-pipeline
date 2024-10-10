import os
import json
import nest_asyncio
from pathlib import Path
from pprint import pprint
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
nest_asyncio.apply()

class SyntheticTestGenerator:
    def __init__(self, openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        # self.directory = directory
        self.documents = []
        self.testset = None


    # Load documents
    def load_documents(self):
        # Clean data
        for filename in os.listdir('src/data'):
            filepath = os.path.join('src/data', filename)

            if os.path.isfile(filepath):
                with open(filepath, 'r') as file:
                    content = file.read()

                modified_content = content.replace('\n', ' ')

                with open(filepath, 'w') as file:
                    file.write(modified_content)

        loader = DirectoryLoader("./src/data")

        self.documents = loader.load()

        for document in self.documents:
            document.metadata['filename'] = document.metadata.get('source', 'unknown_source')

        if self.documents is not None:
            print("Documents successfully loaded.")
        else:
            print("Failed to load documents.")
            exit(1)


    # Generate synthetic test set from loaded documents
    def generate_synthetic_test(self):
        generator_llm = ChatOpenAI(model="gpt-3.5-turbo")
        critic_llm = ChatOpenAI(model="gpt-4o-mini")
        embeddings = OpenAIEmbeddings()

        generator = TestsetGenerator.from_langchain(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings=embeddings
        )

        self.testset = generator.generate_with_langchain_docs(
            documents=self.documents,
            test_size=10,
            distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
        )

        if self.testset is not None:
            print("Testset successfully generated.")
        else:
            print("Failed to generate synthetic tests.")
            exit(1)

    
    # Export results to Panda dataframe
    def visualize_dataframe(self):
        print("Testset exported to Pandas Dataframe.")

        print(self.testset.to_pandas())
