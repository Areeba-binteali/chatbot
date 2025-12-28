from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
import cohere
from qdrant_client import QdrantClient
from agents import enable_verbose_stdout_logging

enable_verbose_stdout_logging()

load_dotenv()
set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=provider
)


# Initialize Cohere client
cohere_client = cohere.Client(os.getenv("COHERE_API"))
# Connect to Qdrant
qdrant = QdrantClient(
   url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API")  
)



def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


@function_tool
def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="physical_ai_textbook",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]



agent = Agent(
    name="Assistant",
    instructions="""
You are an AI tutor for the Physical AI & Humanoid Robotics textbook.

When answering a user question:
1. First, call the `retrieve` tool using the user's query.
2. Use ONLY the information returned by the `retrieve` tool to form your answer.
3. Do NOT use prior knowledge, assumptions, or external information.
4. If the retrieved content does not contain the answer, respond exactly with:
   "I don't know"
5. Do not add explanations, summaries, or extra commentary beyond the answer.

""",
    model=model,
    tools=[retrieve]
)
