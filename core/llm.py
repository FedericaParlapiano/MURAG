from openai import OpenAI

class LLMGenerator:
    """Handles LLM generation via OpenAI-compatible API"""
    
    def __init__(self, model_name: str, base_url: str, api_key: str):
        print(f"🤖 Connecting to API at {base_url} with model: {model_name}")
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 1000, 
                 temperature: float = 0.1, top_p: float = 0.9,
                 json_mode: bool = False, stop: list = None) -> str:
        try:
            kwargs = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt} 
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if stop:
                kwargs["stop"] = stop 
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error communicating with API: {e}"
