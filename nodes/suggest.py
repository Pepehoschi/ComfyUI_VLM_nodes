import folder_paths
import os
from llama_cpp import Llama, LlamaGrammar
from .prompts import system_msg_prompts
from pydantic import BaseModel, Field, validator 
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation
import json
from .prompts import system_msg_prompts
from .prompts import system_msg_simple
from typing import List, Optional
import re
from string import Template
from typing import Any, List
from pydantic import BaseModel, Field, create_model
from typing_extensions import Literal
import torch
import gc
 

supported_LLava_extensions = set(['.gguf'])

try:
    folder_paths.folder_names_and_paths["LLavacheckpoints"] = (folder_paths.folder_names_and_paths["LLavacheckpoints"][0], supported_LLava_extensions)
except:
    # check if LLavacheckpoints exists otherwise create
    if not os.path.isdir(os.path.join(folder_paths.models_dir, "LLavacheckpoints")):
        os.mkdir(os.path.join(folder_paths.models_dir, "LLavacheckpoints"))
        
    folder_paths.folder_names_and_paths["LLavacheckpoints"] = ([os.path.join(folder_paths.models_dir, "LLavacheckpoints")], supported_LLava_extensions)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")

class Analysis(BaseModel):
    """
    Represents entries about an analysis.
    """
    main_character: List[str] = Field(..., description="Description of the main objects of the analysis")
    artform: List[str]  = Field(..., description="List of Artforms of the analysis")
    photo_type: List[str]  = Field(..., description="List of Types of the photo used in the analysis")
    color_with_objects: List[str]  = Field(..., description="List of objects and their colors of the analysis")
    digital_artform: List[str]  = Field(..., description="List of Digital artforms of the analysis")
    background: List[str]  = Field(..., description="List of Background of the analysis") 
    lighting: List[str]  = Field(..., description="List of Lighting settings of the analysis.")

class PromptGen(BaseModel):
    """
    Represents an entry about a prompt.
    """
    prompt : str = Field(..., description="Prompt for the analysis")

class Suggestion(BaseModel):
    """
    Represents an entry about a suggestion.
    """
    suggestion1 : str = Field(..., description="new Suggestion based on the inputs")
    suggestion2 : str = Field(..., description="new Suggestion based on the inputs")
    suggestion3 : str = Field(..., description="new Suggestion based on the inputs")
    suggestion4 : str = Field(..., description="new Suggestion based on the inputs")
    suggestion5 : str = Field(..., description="new Suggestion based on the inputs")

class ArtisticTechniques(BaseModel):
    preferred: List[str] = Field(
        ...,
        description="Long description of Techniques and tools favored for creating the artwork, emphasizing cutting-edge or specialized modern or traditional techniques."
    )
    
    avoided: List[str] = Field(
        ...,
        description="Long description of Techniques and tools favored for creating the artwork, emphasizing cutting-edge or specialized modern or traditional techniques."
    )

class ImageryTheme(BaseModel):
    core_subject: str = Field(
        ...,
        description="Long description of Core subject or theme of the artwork, described vividly to evoke a strong image or emotion."
    )
    additional_elements: Optional[List[str]] = Field(
        default=None,
        description="Long description of Additional elements or motifs to include, enhancing the core theme with specific details or themes for a more immersive and detailed scene."
    )

class VisualStyle(BaseModel):
    desired: List[str] = Field(
        ...,
        description="Long description of Desired visual styles and aesthetic qualities, such as realistic, stylized, or rich artwork."
    )
    undesired: List[str] = Field(
        ...,
        description="Long description of Styles and aesthetic qualities to avoid."
    )


class ArtInspirationNarrative(BaseModel):
    description: str

class ArtPromptSpecification(BaseModel):
    techniques: ArtisticTechniques
    theme: ImageryTheme
    style: VisualStyle
    creative_descriptions: List[ArtInspirationNarrative] = []

    @validator('creative_descriptions', always=True)
    def generate_creative_descriptions(cls, v, values):
        if not values.get('techniques') or not values.get('theme') or not values.get('style'):
            return v  # Ensures prerequisites are met

        # Synthesizing the description
        technique_str = " and ".join(values['techniques'].preferred)
        theme_description = values['theme'].core_subject
        style_description = " and ".join(values['style'].desired)
        additional_elements = ", ".join(values['theme'].additional_elements) if values['theme'].additional_elements else "enriching details"

        # Constructing the integrated creative description
        integrated_description = f"Envision an artwork that utilizes {technique_str}. The essence revolves around '{theme_description}', adorned with {additional_elements}. The visual pursuit should mirror styles such as {style_description}, bringing the concept to life with depth and emotion."

        return [ArtInspirationNarrative(description=integrated_description)]
    
def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def _llama_reasoning_budget(thinking):
    return -1 if _as_bool(thinking, False) else 0

def _llama_temperature(sampling_mode, temperature):
    return _as_float(temperature, 0.2, 0.0, 2.0) if _as_bool(sampling_mode, True) else 0.0

def _as_bool(value, default):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("true", "1", "yes", "on"):
            return True
        if value in ("false", "0", "no", "off"):
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return default

def _as_float(value, default, min_value=None, max_value=None):
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value

def _as_int(value, default, min_value=None, max_value=None):
    if isinstance(value, bool):
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value

def _close_llama_model(llm):
    if llm is None:
        return
    close = getattr(llm, "close", None)
    if callable(close):
        close()

def _release_memory():
    gc.collect()
    try:
        import comfy.model_management
        comfy.model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _llama_text_response(response):
    choice = response["choices"][0]
    if "message" in choice:
        return choice["message"]["content"]
    return choice["text"]

def _clean_llama_text(text):
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    channel_matches = list(re.finditer(r"<\|?channel\|?>|<channel\|>|<\|channel>", text))
    if channel_matches and channel_matches[-1].end() < len(text):
        text = text[channel_matches[-1].end():]

    final_markers = [
        "<|channel|>final<|message|>",
        "<|channel>final<|message>",
        "<channel|>final<message|>",
        "||final",
    ]
    for marker in final_markers:
        if marker in text:
            text = text.rsplit(marker, 1)[-1]

    replacements = [
        "<|im_start|>assistant",
        "<|im_start|>user",
        "<|im_start|>system",
        "<|im_start|>",
        "<|im_end|>",
        "|im_end|>",
        "<|start|>assistant",
        "<|start|>user",
        "<|start|>system",
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|message>",
        "<message|>",
        "<|channel|>thought",
        "<|channel>thought",
        "<channel|>thought",
        "<|channel|>analysis",
        "<|channel>analysis",
        "<channel|>analysis",
        "<|channel|>final",
        "<|channel>final",
        "<channel|>final",
        "||thought",
        "||analysis",
    ]
    for marker in replacements:
        text = text.replace(marker, "")
    text = re.sub(r"<\|?im_[^\s>]*(?:\|?>)?", "", text)
    text = re.sub(r"<?\|?channel\|?>\s*(thought|analysis|final)?", "", text)
    text = re.sub(r"<?channel\|?>\s*(thought|analysis|final)?", "", text)
    text = re.sub(r"<\|?message\|?>", "", text)
    text = re.sub(r"\|\|(thought|analysis|final)", "", text)
    text = re.sub(r"(?im)^\s*(thought|analysis)\s*:\s*", "", text)
    return text.strip()

def _create_llama_text_response(llm, messages, raw_prompt, max_tokens, temperature, top_p, top_k,
                                frequency_penalty, presence_penalty, repeat_penalty, seed=None,
                                sampling_mode=True, min_p=0.05, thinking=False, use_default_template=True):
    if not _as_bool(use_default_template, True):
        messages = [{"role": "user", "content": raw_prompt}]
    common_args = {
        "max_tokens": _as_int(max_tokens, 512, 1),
        "temperature": _llama_temperature(sampling_mode, temperature),
        "top_p": _as_float(top_p, 0.95, 0.0, 1.0),
        "top_k": _as_int(top_k, 40, 0),
        "min_p": _as_float(min_p, 0.05, 0.0, 1.0),
        "frequency_penalty": _as_float(frequency_penalty, 0.0),
        "present_penalty": _as_float(presence_penalty, 0.0),
        "repeat_penalty": _as_float(repeat_penalty, 1.1, 0.0),
        "seed": _as_int(seed, 42, 0) if seed is not None else None,
        "reasoning_budget": _llama_reasoning_budget(thinking),
        "stop": ["<|im_end|>", "|im_end|>", "<|end|>"],
    }
    return llm.create_chat_completion(messages=messages, **common_args)

class PromptGenerateAPI:
    def __init__(self):
        self.session_history = []  
        self.system_msg_prompts = "You are an advanced AI, please assist with the following request."
        self.system_msg_simple = "Simple chat mode activated."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    ["ChatGPT-3.5", "ChatGPT-4", "DeepSeek", "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-35-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-4-0613", "gpt-4-1106-preview", "glm-4"],
                    {
                        "default" : "ChatGPT-3.5"
                    }
                ), 
                "chat_type": 
                    ("BOOLEAN", 
                    {
                        "default": True, "label_on": "PromptGenerator", "label_off": "SimpleChat"
                    }
                ),        
                "api_key": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    }
                ),
                "question": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),                    
                "context_size": (
                    "INT", 
                    {
                        "default": 5, 
                        "min": 0, 
                        "max": 30, 
                        "step": 1
                    }
                ),
                "seed": (
                    "INT", 
                    {
                        "default": 0, 
                        "min": 0, 
                        "max": 0xffffffffffffffff, 
                        "step": 1
                    }
                ),
            },
        }
    RETURN_TYPES = ("STRING",)

    FUNCTION = "generate_prompt"

    CATEGORY = "VLM Nodes/LLM"

    def generate_prompt(self, model_name, chat_type, api_key, description, question, context_size, seed): 
        from openai import OpenAI      
        if chat_type == True:
            system_msg = self.system_msg_prompts
        elif chat_type == False:
            system_msg = self.system_msg_simple


        user_msg = f"""
        Description: {description}
        Optional Question: {question}

        Output: 
        """


        self.session_history = self.session_history[-context_size:]


        messages = [{"role": "system", "content": system_msg}] + self.session_history + [{"role": "user", "content": user_msg}]

        if model_name == "DeepSeek":
            model = "deepseek-chat"
            base_url = "https://api.deepseek.com/v1"
        elif model_name in ["ChatGPT-3.5", "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-35-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]:
            model = "gpt-3.5-turbo"
            base_url = None
        elif model_name in ["ChatGPT-4", "gpt-4-0613", "gpt-4-1106-preview"]:
            model = "gpt-4"
            base_url = None
        elif model_name == "glm-4":
            model = "glm-4"
            base_url = None

        client = OpenAI(api_key=api_key, base_url=base_url)


        completion = client.chat.completions.create(
            model=model,
            messages=messages,  
            seed=seed  
        )

        prompt = completion.choices[0].message.content

        self.session_history += [{"role": "user", "content": user_msg}, {'role': 'assistant', "content": prompt}]

        return (prompt,)
    
class LLMLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
              "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),   
              "max_ctx": ("INT", {"default": 2048, "min": 128, "max": 128000, "step": 64}),
              "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
              "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
              "use_mlock": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                            }
                }
                
    
    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_llm_checkpoint"

    CATEGORY = "VLM Nodes/LLM"
    def load_llm_checkpoint(self, ckpt_name, max_ctx, gpu_layers, n_threads, use_mlock):
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        llm = Llama(model_path = ckpt_path, offload_kqv=True, f16_kv=True, use_mlock=use_mlock, embedding=False, n_batch=1024, last_n_tokens_size=1024, verbose=True, seed=42, n_ctx = max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads,) 
        return (llm, ) 
    
class LLMPromptGenerator:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}), 
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),                             
                "sampling_mode": (["on", "off"], {"default": "on"}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thinking": (["off", "on"], {"default": "off"}),
                "use_default_template": (["on", "off"], {"default": "on"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text_advanced(self,prompt, model, max_tokens, temperature, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty, sampling_mode=True, min_p=0.05, thinking=False, use_default_template=True):
        llm = model
        messages = [
            {"role": "system", "content": system_msg_prompts},
            {"role": "user", "content": prompt},
        ]
        response = _create_llama_text_response(
            llm, messages, prompt, max_tokens, temperature, top_p, top_k,
            frequency_penalty, presence_penalty, repeat_penalty,
            sampling_mode=sampling_mode, min_p=min_p, thinking=thinking,
            use_default_template=use_default_template,
        )
        return (f"{_clean_llama_text(_llama_text_response(response))}", )
    
class LLMSampler:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_msg": ("STRING",{"forceInput": True, "default" : "You are an assistant who perfectly describes images."}),
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}), 
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),
		        "seed": ("INT", {"default": 42, "step": 1}),
                "sampling_mode": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thinking": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "use_default_template": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text_advanced(self, system_msg, prompt, model, max_tokens, temperature, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty, seed, sampling_mode, min_p, thinking, use_default_template):
        llm = model
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        response = _create_llama_text_response(
            llm, messages, prompt, max_tokens, temperature, top_p, top_k,
            frequency_penalty, presence_penalty, repeat_penalty, seed=seed,
            sampling_mode=sampling_mode,
            min_p=min_p,
            thinking=thinking,
            use_default_template=use_default_template,
        )
        return (f"{_clean_llama_text(_llama_text_response(response))}", )

class ChatMusician:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.90, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}), 
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),
		        "seed": ("INT", {"default": 42, "step": 1}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 48000, "step": 1}),
                "sampling_mode": (["on", "off"], {"default": "on"}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thinking": (["off", "on"], {"default": "off"}),
                "use_default_template": (["on", "off"], {"default": "on"}),
            }
        }

    RETURN_NAMES = ("response", "wave_form", "sample_rate", )
    RETURN_TYPES = ("STRING", any, "INT", )
    FUNCTION = "chat_musician"
    CATEGORY = "VLM Nodes/Audio"
    OUTPUT_NODE = True

    def chat_musician(self, prompt, model, max_tokens, temperature, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty, seed, sample_rate, sampling_mode=True, min_p=0.05, thinking=False, use_default_template=True):
        llm = model
        prompt = _parse_text(prompt)
        prompt_template = Template("Human: ${inst} </s> Assistant: ")
        prompt = prompt_template.safe_substitute({"inst": prompt})
        messages = [
            {"role": "user", "content": f"Human: {prompt} </s> Assistant: "},
        ]
        raw_prompt = f"Human: {prompt} </s> Assistant: "
        response = _create_llama_text_response(
            llm, messages, raw_prompt, max_tokens, temperature, top_p, top_k,
            frequency_penalty, presence_penalty, repeat_penalty, seed=seed,
            sampling_mode=sampling_mode, min_p=min_p, thinking=thinking,
            use_default_template=use_default_template,
        )

        from symusic import Score, Synthesizer

        abc_pattern = r'(X:\d+\n(?:[^\n]*\n)+)'
        abc_notation = re.findall(abc_pattern, f"{_clean_llama_text(_llama_text_response(response))}\n")[0]
        s = Score.from_abc(abc_notation)
        audio = Synthesizer().render(s, stereo=True).tolist()[0]
        
        return (abc_notation, audio, sample_rate, )
    
class KeywordExtraction:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),                          
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "keyword_extract"
    CATEGORY = "VLM Nodes/LLM"
    
    def keyword_extract(self, prompt, model, temperature):
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Analysis])
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)


        wrapped_model = LlamaCppAgent(model, debug_output=True,
                                    system_prompt="You are an advanced AI, tasked to create JSON database entries for analysis.\n\n\n" + documentation)

        response = wrapped_model.get_chat_response(prompt, temperature=temperature, grammar=grammar)
        return (response, )
    
class LLavaPromptGenerator:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),                           
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompts"
    CATEGORY = "VLM Nodes/LLM"
    
    def generate_prompts(self, prompt, model, temperature):
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([PromptGen])
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

        wrapped_model = LlamaCppAgent(model, debug_output=True,
                                    system_prompt="You are an advanced AI, tasked to create JSON database entries for creative long prompts for image generation. \n\n\n" + documentation)
        response = wrapped_model.get_chat_response(prompt, temperature=temperature, grammar=grammar, max_tokens=512, repeat_penalty=1.1)
        return (f"{response}", )

class CreativeArtPromptGenerator:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),                           
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "create_creative_art_prompts"
    CATEGORY = "VLM Nodes/LLM"
    
    def create_creative_art_prompts(self, prompt, model, temperature):
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([ArtPromptSpecification])
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

        wrapped_model = LlamaCppAgent(model, debug_output=True,
                                    system_prompt="You are an advanced AI, tasked to create JSON database entries for creative description for image generation. \n\n\n" + documentation)
        response = wrapped_model.get_chat_response(prompt, temperature=temperature, grammar=grammar, max_tokens=512, repeat_penalty=1.1)
        json_response = json.loads(response)
        final_response = json_response["creative_descriptions"][0]["description"]
        return (f"{final_response}", )    

class Suggester:        
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"forceInput": True,"default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
                "randomize": ("BOOLEAN", {"default": True, "label_on": "Consistent", "label_off": "Random"}),                           
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_suggestions"
    CATEGORY = "VLM Nodes/LLM"
    
    def generate_suggestions(self, prompt, model, temperature, randomize):
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Suggestion])
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)
        if randomize: 
            system_msg_suggester = f"You are an advanced AI, tasked to create JSON database entries for generating extremely similar to the prompt. \n\n\n" + documentation
        else:
            #you should suggest variation from prompts
            prompt = "Generate a prompt like: <A random character> <random action> <random place> <random object> <random color>"  
            system_msg_suggester = f"You are an advanced AI, tasked to create JSON database entries for suggesting completely different prompts.. \n\n\n" + documentation 
        wrapped_model = LlamaCppAgent(model, debug_output=True,
                                    system_prompt=system_msg_suggester)
        response = wrapped_model.get_chat_response(prompt, temperature=temperature, grammar=grammar, max_tokens=512, repeat_penalty=1.1)
    
        return (response, )
    

class PydanticAttributeSetter:
    def __init__(self):
        self.attributes = []

    def add_attribute(self, name: str, type_: Any, description: str, categories: List[str] = None):
        if type_ == Literal and categories:
            # Instead of directly using categories, enrich the description to hint at them
            enriched_description = f"For this {description} you should choose from this categories: {', '.join(categories)}."
            enriched_description = enriched_description.replace("  ", " ")
            self.attributes.append((name, str, Field(..., description=enriched_description)))
        else:
            self.attributes.append((name, type_, Field(..., description=description)))

    def create_model(self, model_name: str):
        return create_model(model_name, **{name: (type_, field) for name, type_, field in self.attributes})

class StructuredOutput:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "default": ""}),
                "model": ("CUSTOM", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
                "attribute_name": ("STRING", {"default": ""}),
                "attribute_type": (["str", "int", "float", "bool", "Category"], {"default": "str"}),
                "attribute_description": ("STRING", {"default": ""}),
                "categories": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "keyword_extract"
    CATEGORY = "VLM Nodes/LLM"

    def keyword_extract(self, prompt, model, temperature, attribute_name, attribute_type, attribute_description, categories):
        setter = PydanticAttributeSetter()
        
        if attribute_type == "Category":
            categories = categories.split(",")
            setter.add_attribute(attribute_name, Literal, attribute_description, categories)
        else:
            attribute_type = eval(attribute_type)
            setter.add_attribute(attribute_name, attribute_type, attribute_description)

        Analysis = setter.create_model("Analysis")

        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Analysis])
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

        wrapped_model = LlamaCppAgent(model, debug_output=True,
            system_prompt=f"You are an advanced AI, tasked to create JSON database entries for analysis. \n\n\n{documentation}")

        response = wrapped_model.get_chat_response(prompt, temperature=temperature, grammar=grammar)
        parsed_response = json.loads(response)
        
        return (next(iter(parsed_response.values())),)
    
class LLMOptionalMemoryFreeSimple:
    def __init__(self):
        self.llm = None  # Store the model instance
        self.loaded_config = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
                "max_ctx": ("INT", {"default": 4096, "min": 128, "max": 128000, "step": 64}),
                "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
                "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "prompt": ("STRING", {"forceInput": True}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
                "use_mlock": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text(self, ckpt_name, max_ctx, gpu_layers, n_threads, prompt, temperature, keep_model_loaded, use_mlock):
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        config = (ckpt_path, max_ctx, gpu_layers, n_threads, use_mlock)
        if self.llm is None or self.loaded_config != config:
            _close_llama_model(self.llm)
            self.llm = Llama(model_path=ckpt_path, offload_kqv=True, f16_kv=True, use_mlock=use_mlock, embedding=False, n_batch=1024, last_n_tokens_size=1024, verbose=True, seed=42, n_ctx=max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads, logits_all=True, echo=False)
            self.loaded_config = config

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )

        if not keep_model_loaded:
            _close_llama_model(self.llm)
            self.llm = None
            self.loaded_config = None
            _release_memory()

        return (f"{response['choices'][0]['message']['content']}", )

class LLMOptionalMemoryFreeAdvanced:
    def __init__(self):
        self.llm = None  # Store the model instance
        self.loaded_config = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
                "max_ctx": ("INT", {"default": 4096, "min": 128, "max": 128000, "step": 64}),
                "gpu_layers": ("INT", {"default": 27, "min": 0, "max": 100, "step": 1}),
                "n_threads": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "system_msg": ("STRING", {"default": "You are a helpful AI assistant."}),
                "prompt": ("STRING", {"forceInput": True, "default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),
                "seed": ("INT", {"default": 42, "step": 1}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
                "use_mlock": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "sampling_mode": (["on", "off"], {"default": "on"}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thinking": (["off", "on"], {"default": "off"}),
                "use_default_template": (["on", "off"], {"default": "on"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text_advanced(self, ckpt_name, max_ctx, gpu_layers, n_threads, system_msg, prompt, max_tokens, temperature, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty, seed, keep_model_loaded, use_mlock, sampling_mode=True, min_p=0.05, thinking=False, use_default_template=True):
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        config = (ckpt_path, max_ctx, gpu_layers, n_threads, seed, use_mlock)
        if self.llm is None or self.loaded_config != config:
            _close_llama_model(self.llm)
            self.llm = Llama(model_path=ckpt_path, offload_kqv=True, f16_kv=True, use_mlock=use_mlock, embedding=False, n_batch=1024, last_n_tokens_size=1024, verbose=True, seed=seed, n_ctx=max_ctx, n_gpu_layers=gpu_layers, n_threads=n_threads, logits_all=True, echo=False)
            self.loaded_config = config

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]
        response = _create_llama_text_response(
            self.llm, messages, f"{system_msg}\n\n{prompt}", max_tokens, temperature, top_p, top_k,
            frequency_penalty, presence_penalty, repeat_penalty, seed=seed,
            sampling_mode=sampling_mode, min_p=min_p, thinking=thinking,
            use_default_template=use_default_template,
        )

        if not keep_model_loaded:
            _close_llama_model(self.llm)
            self.llm = None
            self.loaded_config = None
            _release_memory()

        return (f"{_clean_llama_text(_llama_text_response(response))}", )


NODE_CLASS_MAPPINGS = {
    "LLMLoader": LLMLoader,
    "LLMSampler": LLMSampler,
    "LLMPromptGenerator": LLMPromptGenerator,
    "KeywordExtraction": KeywordExtraction,
    "LLavaPromptGenerator": LLavaPromptGenerator,
    "Suggester": Suggester,
    "PromptGenerateAPI": PromptGenerateAPI,
    "CreativeArtPromptGenerator": CreativeArtPromptGenerator,
    "ChatMusician": ChatMusician,
    "StructuredOutput": StructuredOutput,
    "LLMOptionalMemoryFreeSimple": LLMOptionalMemoryFreeSimple,
    "LLMOptionalMemoryFreeAdvanced": LLMOptionalMemoryFreeAdvanced,
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMLoader": "LLMLoader",
    "LLMSampler": "LLMSampler",
    "LLMPromptGenerator": "LLM PromptGenerator",
    "KeywordExtraction": "Get Keywords",
    "LLavaPromptGenerator": "LLava PromptGenerator",
    "Suggester": "Suggester",
    "PromptGenerateAPI": "API PromptGenerator",
    "CreativeArtPromptGenerator": "Creative Art PromptGenerator",
    "ChatMusician": "ChatMusician",
    "StructuredOutput": "Structured Output",
    "LLMOptionalMemoryFreeSimple": "LLM Simple (Memory Optional)",
    "LLMOptionalMemoryFreeAdvanced": "LLM Advanced (Memory Optional)",
}
