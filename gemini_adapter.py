import os, itertools, json
from google import genai
from google.genai.types import FunctionDeclaration, GenerationConfig
from llm_adapter import BaseAdapter, LLMChunk
from tool_discovery import get_function_declarations, execute_tool_sync

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

class GeminiAdapter(BaseAdapter):
    MODEL = config.get('GEMINI_MODEL', 'gemini-2.5-flash-preview-05-20')
    
    def _client(self):
        return genai.Client()
    
    def generate(self, messages):
        tools = get_function_declarations()
        stream = self._client().models.generate_content_stream(
            model=self.MODEL,
            contents=messages,
            tools=tools,
            generation_config=GenerationConfig(
                temperature=config.get('GEMINI_TEMPERATURE', 0.7),
                top_p=config.get('GEMINI_TOP_P', 0.9),
                max_output_tokens=config.get('GEMINI_MAX_OUTPUT_TOKENS', 4096),
            )
        )
        for chunk in stream:
            for part in chunk.candidates[0].content.parts:
                if hasattr(part, 'text'):
                    yield LLMChunk(role='assistant', text=part.text)
                elif hasattr(part, 'function_call'):
                    result = execute_tool_sync(part.function_call.name, part.function_call.args)
                    stream.send_tool_response(result)
                yield # What do we yield here?
