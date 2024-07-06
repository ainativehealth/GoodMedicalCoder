import os
from dotenv import load_dotenv
import instructor
from openai import  AzureOpenAI

load_dotenv()

# Azure OpenAI
azure_client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2024-02-15-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],

   )
azure_client = instructor.patch(azure_client)


