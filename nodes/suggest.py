import folder_paths
import os
from llama_cpp import LlamaGrammar
from .prompts import system_msg_prompts
from pydantic import BaseModel, Field, validator 
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation
import json
from .prompts import system_msg_prompts
from .prompts import system_msg_simple
from typing import List, Optional
import re
import asyncio
from string import Template
from typing import Any, List
from pydantic import BaseModel, Field, create_model
from typing_extensions import Literal
from aiohttp import web
from server import PromptServer
from .llama_manager import (
    get_loaded_llama_by_path,
    get_managed_llama,
    loaded_llama_statuses,
    llama_config,
    managed_status,
    managed_for_llama,
)
 

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
    managed = managed_for_llama(llm)
    if managed is not None:
        with managed.lock:
            return llm.create_chat_completion(messages=messages, **common_args)
    return llm.create_chat_completion(messages=messages, **common_args)


def _generate_managed_llama_response(managed, system_msg, prompt, max_tokens, temperature, top_p, top_k,
                                     frequency_penalty, presence_penalty, repeat_penalty, seed,
                                     sampling_mode=True, min_p=0.05, thinking=False,
                                     use_default_template=True):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    with managed.lock:
        response = _create_llama_text_response(
            managed.llm,
            messages,
            f"{system_msg}\n\n{prompt}",
            max_tokens,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            seed=seed,
            sampling_mode=sampling_mode,
            min_p=min_p,
            thinking=thinking,
            use_default_template=use_default_template,
        )
    return _clean_llama_text(_llama_text_response(response))


def _scratchpad_send(payload):
    loader_config = payload.get("loader_config") or {}
    ckpt_name = loader_config.get("ckpt_name") or payload.get("ckpt_name")
    if not ckpt_name:
        return {"ok": False, "error": "No checkpoint selected."}

    ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
    if not ckpt_path:
        return {"ok": False, "error": f"Checkpoint not found: {ckpt_name}"}

    seed = _as_int(payload.get("seed"), 42, 0)
    managed = get_loaded_llama_by_path(ckpt_path)
    if loader_config:
        config = llama_config(
            ckpt_path,
            _as_int(loader_config.get("max_ctx"), 2048, 128),
            _as_int(loader_config.get("gpu_layers"), 10, 0),
            _as_int(loader_config.get("n_threads"), 8, 1),
            42,
            _as_bool(loader_config.get("use_mlock"), False),
        )
        managed = get_managed_llama(config, auto_load=True)

    model_status = {
        "source": "loader_config" if loader_config else ("reused_loaded_model_by_path" if managed is not None else "not_loaded"),
        "requested_checkpoint": ckpt_path,
        "loader_node_id": payload.get("loader_node_id"),
        "selected_config": managed_status(managed) if managed is not None else None,
        "loaded_models": loaded_llama_statuses(),
    }
    if managed is None:
        return {
            "ok": False,
            "error": "No loaded model found for this checkpoint. Connect LLM Model Loader to the Scratchpad model input or run the loader once first.",
            "model_status": model_status,
        }

    text = _generate_managed_llama_response(
        managed,
        payload.get("system_msg") or "You are a helpful AI assistant.",
        payload.get("prompt") or "",
        payload.get("max_tokens"),
        payload.get("temperature"),
        payload.get("top_p"),
        payload.get("top_k"),
        payload.get("frequency_penalty"),
        payload.get("presence_penalty"),
        payload.get("repeat_penalty"),
        seed,
        sampling_mode=payload.get("sampling_mode", True),
        min_p=payload.get("min_p", 0.05),
        thinking=payload.get("thinking", False),
        use_default_template=payload.get("use_default_template", True),
    )
    model_status["loaded_models"] = loaded_llama_statuses()
    return {"ok": True, "text": text, "model_loaded": True, "model_status": model_status}


@PromptServer.instance.routes.post("/vlmnodes/llm_scratchpad/send")
async def vlmnodes_llm_scratchpad_send(request):
    try:
        payload = await request.json()
        result = await asyncio.to_thread(_scratchpad_send, payload)
        status = 200 if result.get("ok") else 400
        return web.json_response(result, status=status)
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)

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
    
class LLMCheckpointSelector:
    DESCRIPTION = (
        "Selects a GGUF checkpoint from the LLavacheckpoints model folder. "
        "Use this only to feed the LLM Model Loader checkpoint input."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), {
                    "tooltip": "Checkpoint name from the models/LLavacheckpoints folder."
                }),
            }
        }

    RETURN_TYPES = (any, "STRING")
    RETURN_NAMES = ("ckpt_name", "ckpt_name_string")
    FUNCTION = "select_checkpoint"
    CATEGORY = "VLM Nodes/LLM"

    def select_checkpoint(self, ckpt_name):
        return (ckpt_name, ckpt_name)


class LLMLoader:
    DESCRIPTION = (
        "Canonical llama.cpp text model loader. This node owns checkpoint and load arguments "
        "such as context size, GPU layers, threads, and mlock. Downstream LLM nodes should consume its model output."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
              "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), {
                  "tooltip": "GGUF checkpoint to load from models/LLavacheckpoints."
              }),
              "max_ctx": ("INT", {
                  "default": 2048,
                  "min": 128,
                  "max": 128000,
                  "step": 64,
                  "tooltip": "llama.cpp context size. This is part of the loaded model resource config."
              }),
              "gpu_layers": ("INT", {
                  "default": 10,
                  "min": 0,
                  "max": 100,
                  "step": 1,
                  "tooltip": "Number of layers offloaded to GPU. Lower this if the model fails with CUDA OOM."
              }),
              "n_threads": ("INT", {
                  "default": 8,
                  "min": 1,
                  "max": 100,
                  "step": 1,
                  "tooltip": "CPU threads used by llama.cpp."
              }),
              "use_mlock": ("BOOLEAN", {
                  "default": False,
                  "label_on": "On",
                  "label_off": "Off",
                  "tooltip": "Ask llama.cpp to keep model pages locked in RAM when supported."
              }),
                            }
                }
                
    
    RETURN_TYPES = ("CUSTOM", "STRING")
    RETURN_NAMES = ("model", "status_json")
    FUNCTION = "load_llm_checkpoint"

    CATEGORY = "VLM Nodes/LLM"
    def load_llm_checkpoint(self, ckpt_name, max_ctx, gpu_layers, n_threads, use_mlock):
        ckpt_path = folder_paths.get_full_path("LLavacheckpoints", ckpt_name)
        config = llama_config(ckpt_path, max_ctx, gpu_layers, n_threads, 42, use_mlock)
        managed = get_managed_llama(config, auto_load=True)
        status = {
            "source": "loaded_or_reused",
            "selected_config": managed_status(managed),
            "loaded_models": loaded_llama_statuses(),
        }
        return (managed.llm, json.dumps(status, indent=2))
    
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
    DESCRIPTION = "Simple text generation using a model from LLM Model Loader. Load settings live only on the loader."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CUSTOM", {"default": ""}),
                "prompt": ("STRING", {"forceInput": True}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text(self, model, prompt, temperature):
        llm = model
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        response = _create_llama_text_response(
            llm,
            messages,
            prompt,
            512,
            temperature,
            0.95,
            40,
            0.0,
            0.0,
            1.1,
        )

        return (f"{response['choices'][0]['message']['content']}", )

class LLMOptionalMemoryFreeAdvanced:
    DESCRIPTION = "Advanced text generation using a model from LLM Model Loader. Load settings live only on the loader."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CUSTOM", {"default": ""}),
                "system_msg": ("STRING", {"forceInput": True, "default": "You are a helpful AI assistant."}),
                "prompt": ("STRING", {"forceInput": True, "default": ""}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "step": 1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),
                "seed": ("INT", {"default": 42, "step": 1}),
                "sampling_mode": (["on", "off"], {"default": "on"}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thinking": (["off", "on"], {"default": "off"}),
                "use_default_template": (["on", "off"], {"default": "on"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text_advanced"
    CATEGORY = "VLM Nodes/LLM"

    def generate_text_advanced(self, model, system_msg, prompt, max_tokens, temperature, top_p, top_k, frequency_penalty, presence_penalty, repeat_penalty, seed, sampling_mode=True, min_p=0.05, thinking=False, use_default_template=True):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        response = _create_llama_text_response(
            model,
            messages,
            f"{system_msg}\n\n{prompt}",
            max_tokens,
            temperature,
            top_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            seed=seed,
            sampling_mode=sampling_mode,
            min_p=min_p,
            thinking=thinking,
            use_default_template=use_default_template,
        )
        return (_clean_llama_text(_llama_text_response(response)), )


class LLMScratchpad:
    DESCRIPTION = (
        "Scratchpad chat UI that reuses an already loaded LLM Model Loader instance by checkpoint. "
        "It does not own llama.cpp load settings and will not auto-load another model copy."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CUSTOM", {"default": ""}),
                "ckpt_name": (folder_paths.get_filename_list("LLavacheckpoints"), ),
                "system_msg": ("STRING", {"default": "You are a helpful AI assistant.", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "step": 1}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "step": 0.01}),
                "seed": ("INT", {"default": 42, "step": 1}),
                "sampling_mode": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
                "thinking": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "use_default_template": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
                "response": ("STRING", {"default": "", "multiline": True}),
                "model_status": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "output_response"
    CATEGORY = "VLM Nodes/LLM"

    def output_response(self, model, ckpt_name, system_msg, prompt, max_tokens, temperature, top_p, top_k, min_p,
                        frequency_penalty, presence_penalty, repeat_penalty, seed, sampling_mode,
                        thinking, use_default_template, response, model_status=""):
        return (response or "", )


NODE_CLASS_MAPPINGS = {
    "LLMCheckpointSelector": LLMCheckpointSelector,
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
    "LLMScratchpad": LLMScratchpad,
}
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMCheckpointSelector": "LLM Checkpoint Selector",
    "LLMLoader": "LLM Model Loader",
    "LLMSampler": "LLMSampler",
    "LLMPromptGenerator": "LLM PromptGenerator",
    "KeywordExtraction": "Get Keywords",
    "LLavaPromptGenerator": "LLava PromptGenerator",
    "Suggester": "Suggester",
    "PromptGenerateAPI": "API PromptGenerator",
    "CreativeArtPromptGenerator": "Creative Art PromptGenerator",
    "ChatMusician": "ChatMusician",
    "StructuredOutput": "Structured Output",
    "LLMOptionalMemoryFreeSimple": "LLM Simple",
    "LLMOptionalMemoryFreeAdvanced": "LLM Advanced",
    "LLMScratchpad": "LLM Scratchpad",
}
