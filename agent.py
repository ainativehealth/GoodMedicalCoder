from pydantic import BaseModel
from config import groq_client#, azure_client #, openai_client
from langfuse.decorators import observe
class Agent(BaseModel):
    response_model: type[BaseModel]
    ai_provider: str
    model: str

    def __init__(self, response_model, ai_provider, model):
        super().__init__(response_model=response_model,
                         ai_provider=ai_provider,
                         model=model)

    def inference(self, message: str, system_prompt: str):
        client = self._get_client(sync=True)
        return self._perform_inference(client, message, system_prompt)

    def _get_client(self, sync=True):
        if sync:
            if self.ai_provider == "azure_client":
                return azure_client
            if self.ai_provider == "openai_client":
                return openai_client
            if self.ai_provider == "groq_client":
                return groq_client
            else:
                raise ValueError(f"Invalid AI provider: {self.ai_provider}")
        else:
            raise ValueError(f"Invalid AI provider: {self.ai_provider}")

    @observe()
    def _perform_inference(self, client, message: str, system_prompt: str):
        try:

            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": message
                }],
                temperature=0,
                # response_model=self.response_model,
                max_tokens=8000,
                max_retries=2,
            )
            # print('@@@@@@@@@@@@', response.choices[0])
            # response_dict = response.choices[0].message.content
            response_dict = response.dict() if isinstance(response, BaseModel) else response
            return response_dict

        except Exception as e:
            raise RuntimeError(f"Inference failed with {self.model}: {str(e)}")
