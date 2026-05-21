import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM
from config import Config

class TextEncoderWrapper(nn.Module):
    def __init__(self, model_id=None, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.model_id = model_id or Config.text_model_id
        self.dtype = dtype
        self.device = device
        
        print(f"[TextEncoder] Loading configuration: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        
        self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self.is_causal = True
        
        if getattr(self.config, "is_encoder_decoder", False) or "t5" in self.model_id.lower() or "t5gemma" in self.model_id.lower():
            self.is_causal = False

        if not self.is_causal:
            print("[TextEncoder] Loading Seq2Seq model via AutoModelForSeq2SeqLM...")
            self.text_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                torch_dtype=dtype
            )
        else:
            print("[TextEncoder] Loading Causal model via AutoModel...")
            self.text_model = AutoModel.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                torch_dtype=dtype
            )
        
        self.text_model.eval()
        self.text_model.to(device)
        
        try:
            if hasattr(self.config, "hidden_size"):
                self.hidden_size = self.config.hidden_size
            elif hasattr(self.config, "encoder") and hasattr(self.config.encoder, "hidden_size"):
                self.hidden_size = self.config.encoder.hidden_size
            elif (hasattr(self.config, "encoder") and 
                  hasattr(self.config.encoder, "text_config") and 
                  hasattr(self.config.encoder.text_config, "hidden_size")):
                self.hidden_size = self.config.encoder.text_config.hidden_size
            else:
                raise AttributeError()
            print(f"[TextEncoder] Hidden size detected: {self.hidden_size}")
        except:
            self.hidden_size = getattr(Config, 'text_embed_dim', 2048)
            print(f"[TextEncoder] Could not detect hidden size, using fallback: {self.hidden_size}")

    @torch.no_grad()
    def encode(self, prompts, max_length=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        max_length = max_length or Config.max_token_length
        
        inputs = self.tokenizer(prompts, max_length=max_length, padding="max_length", truncation=True,
                                return_tensors="pt").to(self.device)

        if not self.is_causal:
            # Extract states using the standard .get_encoder() API compatible with all Seq2Seq classes
            encoder = self.text_model.get_encoder()
            outputs = encoder(**inputs, output_hidden_states=True)
            hidden = outputs.last_hidden_state
        else:
            # Causal Decoder (Qwen3): Extract representations across trailing layers
            outputs = self.text_model(**inputs, output_hidden_states=True)
            hidden = torch.stack(outputs.hidden_states[-4:]).mean(dim=0)
        
        embeds = hidden
        
        return embeds, inputs.attention_mask.bool()

    def forward(self, prompts):
        embeds, _ = self.encode(prompts)
        return embeds