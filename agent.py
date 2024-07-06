from pydantic import BaseModel
from config import azure_client #, openai_client

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
            else:
                raise ValueError(f"Invalid AI provider: {self.ai_provider}")
        else:
            raise ValueError(f"Invalid AI provider: {self.ai_provider}")

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
                response_model=self.response_model,
                max_tokens=2000,
                max_retries=2,
            )
            response_dict = response.dict() if isinstance(response, BaseModel) else response
            return response_dict

        except Exception as e:
            raise RuntimeError(f"Inference failed with {self.model}: {str(e)}")
