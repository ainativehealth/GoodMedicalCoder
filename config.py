import os
from dotenv import load_dotenv
import instructor
# from openai import  AzureOpenAI, OpenAI
from langfuse.openai import OpenAI

load_dotenv()

# openai_client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])
# openai_client = instructor.patch(openai_client)

groq_client = OpenAI(api_key = os.environ["GROQ_API_KEY"], base_url=os.environ["GROQ_ENDPOINT"], timeout=120 )
groq_client = instructor.patch(groq_client)

# groq_client = OpenAI(api_key ="ollama", base_url="http://localhost:11434/v1", timeout=120 )
# groq_client = instructor.patch(groq_client)

# azure_client = AzureOpenAI(
#         api_key=os.environ["AZURE_OPENAI_API_KEY"],
#         api_version="2024-02-15-preview",
#         azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#         azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],

#    )
# azure_client = instructor.patch(azure_client)
