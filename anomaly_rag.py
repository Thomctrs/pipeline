from typing import List, Dict, Union
import numpy as np
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel
from dotenv import load_dotenv
import os

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
        self.reset_pipeline()
        load_dotenv()
        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME"),
                "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
                "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            }
        )

    def reset_pipeline(self):
        self.documents = None
        self.index = None
        self.anomaly_data = {}
        self.conversation_state = "start"

    async def on_startup(self):
        self.embedding_model = FastEmbedEmbeddings(model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME)
        self.llm = ChatOllama(model_name=self.valves.LLAMAINDEX_MODEL_NAME)

    async def on_shutdown(self):
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
        embeddings = [self.embedding_model.embed(text) for text in texts]
        return np.array(embeddings)

    def recalculate_embeddings_for_test_anomalies(self, test_anomaly_title, test_anomaly_description):
        combined_text = f"{test_anomaly_title} {test_anomaly_description}"
        test_embedding = self.embedding_model.embed(combined_text)
        return np.array([test_embedding])

    def get_most_similar_anomalies(self, test_anomaly_embedding, anomaly_embeddings):
        similarities = cosine_similarity(test_anomaly_embedding, anomaly_embeddings)
        most_similar_indices = np.argsort(-similarities[0])[:3]
        return most_similar_indices, similarities[0]

    def generate_recommendation_text(self, similar_anomalies, problem):
        prompt = f"The user has encountered a problem described as follows:\n\nTitle: {problem['title']}\nDescription: {problem['abstract']}\n\nBased on this problem, here are descriptions of 3 similar anomalies:\n\n"
        for idx, anomaly in enumerate(similar_anomalies):
            prompt += f"Anomaly {idx + 1}:\nTitle: {anomaly['title']}\nDescription: {anomaly['description']}\n\n"
        prompt += "Please provide a detailed recommendation to solve the user's problem based on the similarities with these anomalies."

        response = self.llm.generate(prompt=prompt)
        return response.strip()

    def get_next_question(self) -> str:
        prompts = {
            "start": "Welcome to the anomaly reporting system. Please provide the title of the anomaly you encountered.",
            "ask_title": "Thank you! Can you now provide a brief description of the anomaly?",
            "ask_abstract": "Got it. Now, please provide the anomaly number.",
            "ask_number": "Thanks. Finally, do you have any additional comments or details about the anomaly?",
            "ask_comment": "We have gathered all the necessary information. Do you want to confirm the details?\n\nTitle: {self.anomaly_data.get('title')}\nDescription: {self.anomaly_data.get('abstract')}\nNumber: {self.anomaly_data.get('number')}\nComment: {self.anomaly_data.get('comment')}\n\nIs this information correct? (yes/no)"
        }
        return prompts.get(self.conversation_state, "Thank you for using our system. Please restart the process if you need to.")

    def handle_llm_interaction(self, user_input: str) -> str:
        prompt = f"{self.get_next_question()}\nUser Response: {user_input}\n\nWhat should be the next question or action?"
        response = self.llm.generate(prompt=prompt)
        return response.strip()
 
    def process_user_response(self, user_input: str) -> str:
        if self.conversation_state == "confirmation":
            if user_input.lower() in ["yes", "y"]:
                self.conversation_state = "finished"
                return "Thanks! Processing the anomaly data."
            else:
                self.conversation_state = "ask_title"
                self.anomaly_data = {}
                return "It seems there was an issue. Let's start over. Please provide the title of the anomaly."

        if user_input.lower() in ["reset"]:
            self.reset_pipeline()
            return "The conversation has been reset. Please provide the title of the anomaly to start over."

        return self.handle_llm_interaction(user_input)

    def pipe(self, user_message: str, model_id: str, messages: List[Dict], body: Dict) -> Union[str, None]:
        if self.conversation_state != "finished":
            return self.process_user_response(user_message)

        problem = {
            'title': self.anomaly_data['title'],
            'abstract': self.anomaly_data['abstract'],
            'number': self.anomaly_data['number'],
            'comment': self.anomaly_data['comment']
        }

        # Connect to Neo4j and retrieve anomalies
        driver = self.neo4j_connection()
        try:
            data = self.get_anomalies(driver)
        finally:
            driver.close()

        # Clean the data
        cleaned_anomaly_data = self.clean_text(data)
        if not cleaned_anomaly_data:
            return "No valid anomaly data found after cleaning."

        combined_texts = [f"{item['title']} {item['description']}" for item in cleaned_anomaly_data]
        anomaly_embeddings = self.compute_embeddings_for_combined_texts(combined_texts)

        test_anomaly_embeddings = self.recalculate_embeddings_for_test_anomalies(
            problem['title'], problem['abstract']
        )

        # Find most similar anomalies
        most_similar_indices, similarity_scores = self.get_most_similar_anomalies(test_anomaly_embeddings, anomaly_embeddings)

        top3_anomalies = [
            {"title": cleaned_anomaly_data[idx]['title'],
             "description": cleaned_anomaly_data[idx]['description'],
             "similarity_score": similarity_scores[idx]}
            for idx in most_similar_indices
        ]

        # Generate recommendation
        recommendation_text = self.generate_recommendation_text(top3_anomalies, problem)

        # Return the result
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

        self.reset_pipeline()
        return result

    class AnomalyRetrievalAndRecommendationPipeline:
        @staticmethod
        async def run_pipeline(user_message: str, model_id: str, messages: List[Dict], body: Dict) -> str:
            pipeline = Pipeline()
            await pipeline.on_startup()
            pipe_result = pipeline.pipe(user_message, model_id, messages, body)
            await pipeline.on_shutdown()
            return pipe_result
