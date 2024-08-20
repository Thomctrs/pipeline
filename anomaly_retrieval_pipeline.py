from typing import List, Union, Generator, Iterator
import json
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import openai   
import os
from pydantic import BaseModel
from dotenv import load_dotenv



class Pipeline:
    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        NEO4J_URI: str
        NEO4J_USER: str
        NEO4J_PASSWORD: str
        OPENAI_API_KEY: str

    def __init__(self):
        self.documents = None
        self.index = None
        load_dotenv()
        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3_8b"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
                "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            } 
        )

        openai.api_key = self.valves.OPENAI_API_KEY
        self.conversation_state = "original_query"
        self.anomaly_data = {}

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        global documents, index

        # Optionally, load documents if needed for LlamaIndex
        #self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        #self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # Actions to perform on server shutdown
        pass

    def neo4j_connection(self):
        return GraphDatabase.driver(self.valves.NEO4J_URI, auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD))

    def get_anomalies(self, driver):
        query = """
        MATCH (fa:FicheAnomalie)
        RETURN fa.fan_description_anomalie AS description, fa.fan_intitule AS intitule
        """
        with driver.session() as session:
            result = session.run(query)
            return [{"title": record["intitule"], "description": record["description"]} for record in result]

    def clean_text(self, data):
        return [item for item in data if isinstance(item['description'], str) and isinstance(item['title'], str)]

    def compute_embeddings_for_combined_texts(self, texts):
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        n_components = min(100, X.shape[1])
        svd = TruncatedSVD(n_components=n_components)
        reduced_X = svd.fit_transform(X)
        return reduced_X, vectorizer, svd

    def recalculate_embeddings_for_test_anomalies(self, test_anomaly_title, test_anomaly_description, vectorizer, svd):
        combined_text = [f"{test_anomaly_title} {test_anomaly_description}"]
        X_test = vectorizer.transform(combined_text)
        return svd.transform(X_test)

    def get_most_similar_anomalies(self, test_anomaly_embedding, anomaly_embeddings):
        similarities = cosine_similarity(test_anomaly_embedding, anomaly_embeddings)
        most_similar_indices = np.argsort(-similarities[0])[:3]
        return most_similar_indices, similarities[0]

    def generate_recommendation_text(self, similar_anomalies, problem):
        prompt = f"The user has encountered a problem described as follows:\n\nTitle: {problem['title']}\nDescription: {problem['abstract']}\n\nBased on this problem, here are descriptions of 3 similar anomalies:\n\n"
        for idx, anomaly in enumerate(similar_anomalies):
            prompt += f"Anomaly {idx + 1}:\nTitle: {anomaly['title']}\nDescription: {anomaly['description']}\n\n"
        prompt += "Please provide a detailed recommendation to solve the user's problem based on the similarities with these anomalies."
        
        openai.api_key = self.valves.OPENAI_API_KEY

        response = openai.chat.completions.create(
           
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at providing solutions for software anomalies."},
                {"role": "user", "content": prompt}
            ],
            

        )

        return response.choices[0].message.content.strip()
    

    def ask_next_question(self):
        if self.conversation_state == "original_query":
            return "Welcome to the anomaly reporting system. Please provide the details of the anomaly you encountered"
        if self.conversation_state == "ask_title":
            return f"Ok let's start ! Please provide the title of the anomaly. {self.conversation_state}"
        elif self.conversation_state == "ask_abstract":
            return f"Please provide a brief abstract of the anomaly.{self.conversation_state}"
        elif self.conversation_state == "ask_number":
            return f"Now, Please provide the anomaly number. {self.conversation_state}"
        elif self.conversation_state == "ask_comment":
            return "Please provide any additional comments about the anomaly."
        elif self.conversation_state == "confirmation":
            return f"Here is the summary of the anomaly:\n\nTitle: {self.anomaly_data['title']}\nAbstract: {self.anomaly_data['abstract']}\nNumber: {self.anomaly_data['number']}\nComment: {self.anomaly_data['comment']}\n\nIs this information correct? (yes/no)"
        else:
            return "Thank you! Your anomaly has been recorded. Let me find the most similar anomalies for you. \n Please enter to proceed"

    def process_user_response(self, user_input):
        user_input = user_input.strip()
        print(f"State before processing: {self.conversation_state}, User input: {user_input}")

        if self.conversation_state == "original_query":
            self.anomaly_data['original_query'] = user_input
            self.conversation_state = "ask_title"
        elif self.conversation_state == "ask_title":
            print("ask_title")
            self.anomaly_data['title'] = user_input
            self.conversation_state = "ask_abstract"
        elif self.conversation_state == "ask_abstract":
            print("ask_abstract")
            self.anomaly_data['abstract'] = user_input
            self.conversation_state = "ask_number"
        elif self.conversation_state == "ask_number":
            print("ask_number")
            self.anomaly_data['number'] = user_input
            self.conversation_state = "ask_comment"
        elif self.conversation_state == "ask_comment":
            self.anomaly_data['comment'] = user_input
            self.conversation_state = "confirmation"
        elif self.conversation_state == "confirmation":
            if user_input.lower() in ["yes", "y"]:
                self.conversation_state = "finished"
            else:
                self.conversation_state = "original_query"
                return "Let's start. Please provide the title of the new anomaly."

        return self.ask_next_question()

    def handle_conversation(self, user_input):
        return self.process_user_response(user_input)

    
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        if self.conversation_state != "finished":
            return self.handle_conversation(user_message)

        # Assuming the user_message contains the problem description
        problem = {
            'original_query': self.anomaly_data['original_query'],
            'title': self.anomaly_data['title'],
            'abstract': self.anomaly_data['abstract'],
            'number': self.anomaly_data['number'],
            'comment': self.anomaly_data['comment']
        }

        # Step 1: Connect to Neo4j and retrieve anomalies
        driver = self.neo4j_connection()
        try:
            data = self.get_anomalies(driver)
        finally:
            driver.close()

        # Step 2: Process and clean the data
        cleaned_anomaly_data = self.clean_text(data)
        if not cleaned_anomaly_data:
            return "No valid anomaly data found after cleaning."

        combined_texts = [f"{item['title']} {item['description']}" for item in cleaned_anomaly_data]
        anomaly_embeddings, vectorizer, svd = self.compute_embeddings_for_combined_texts(combined_texts)

        test_anomaly_embeddings = self.recalculate_embeddings_for_test_anomalies(
            problem['title'], problem['abstract'], vectorizer, svd
        )

        # Step 3: Find most similar anomalies
        most_similar_indices, similarity_scores = self.get_most_similar_anomalies(test_anomaly_embeddings, anomaly_embeddings)

        top3_anomalies = [
            {
                "title": cleaned_anomaly_data[idx]["title"],
                "description": cleaned_anomaly_data[idx]["description"],
                "similarity_score": similarity_scores[idx],
            }
            for idx in most_similar_indices
        ]

        openai.api_key = self.valves.OPENAI_API_KEY

        # Step 4: Generate recommendation
        recommendation_text = self.generate_recommendation_text(top3_anomalies, problem)
        
        # Step 5: Return the result
        result = (
        "🛠 **Problem Details** 🛠\n"
        f"   - **Title**: {problem['title']}\n"
        f"   - **Abstract**: {problem['abstract']}\n"
        f"   - **Number**: {problem['number']}\n"
        f"   - **Comment**: {problem['comment']}\n\n"
        
        "🔍 **Top 3 Similar Anomalies** 🔍\n"
        )

        for i, anomaly in enumerate(top3_anomalies, 1):
            result += (
                f"   {i}. **Anomaly**\n"
                f"      - **Title**: {anomaly['title']}\n"
                f"      - **Description**: {anomaly['description']}\n"
                f"      - **Similarity Score**: {anomaly['similarity_score']:.2f}\n\n"
            )

        result += (
            "💡 **Recommendation to solve the issue** 💡\n"
            f"{recommendation_text}\n"
        )

        return result

    
    class AnomalyRetrievalAndRecommendationPipeline:
        @staticmethod
        async def run_pipeline(user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
     
            pipeline = Pipeline()
            await pipeline.on_startup()
            return pipeline.pipe(user_message, model_id, messages, body)
