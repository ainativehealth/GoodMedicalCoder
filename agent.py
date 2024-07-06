from pydantic import BaseModel
from datetime import datetime
from config import  azure_client

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
            else:
                raise ValueError(f"Invalid AI provider: {self.ai_provider}")
        else:
            raise ValueError(f"Invalid AI provider: {self.ai_provider}")

    def _perform_inference(self, client, message: str, system_prompt: str):
        with logfire.span(f"Performing inference with {self.model}",
                          model=self.model,
                          system_prompt=system_prompt,
                          message=message) as span:
            try:
                if self.ai_provider == "gemini_client":
                    response = client.chat.completions.create(
                        messages=[{
                            "role": "system",
                            "content": system_prompt
                        }, {
                            "role": "user",
                            "content": message
                        }],
                        response_model=self.response_model,
                        max_tokens=2000,
                    )
                else:
                    print(client)
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
                    response_dict = response.dict() if isinstance(
                        response, BaseModel) else response
                    logfire.info(f"Inference completed with {self.model}",
                                 model=self.model,
                                 system_prompt=system_prompt,
                                 message=message,
                                 response=response_dict)
                    return response_dict

                response_dict = response.dict() if isinstance(
                    response, BaseModel) else response
                span.set_attribute("response", response_dict)
                logfire.info(f"Inference completed with {self.model}",
                             model=self.model,
                             system_prompt=system_prompt,
                             message=message,
                             response=response_dict)
                return response_dict

            except Exception as e:
                logfire.error(f"Inference failed with {self.model}: {str(e)}",
                              model=self.model,
                              system_prompt=system_prompt,
                              message=message)
                raise
