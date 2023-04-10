import torch
from torch import nn
import numpy as np
from typing import Optional, List, Dict, Union

from transformers import pipeline, AutoTokenizer

from .base_model import BaseModel
from .model_utils import _top_k_logits, _top_p_logits

from transformers import CLIPVisionModel



SUPPORTED_LMS = ['distilgpt2', 'gpt2', 'gpt2-medium',
                 'gpt2-large', 'gpt2-xl']

LM_HIDDEN_SIZES = {'distilgpt2': 768,
                   'gpt2': 768,
                   'gpt2-medium': 1024,
                   'gpt2-large': 1280,
                   'gpt2-xl': 1600}


class ImagePromptModel(BaseModel):
    """
    Uses an Cross Attention Module to make the hidden states of an pre-trained LM conditioned on the CLIP encoded image.

    The modified hidden state can then be passed into the original LM head to obtain output token logits. 
    """
    def __init__(
        self,
        # MLP-specific parameters
        policy_lm: str,
        hidden_size: int,
        logit_bias: float,
        fluent: bool,
        fluent_top_k: Optional[int],
        # Generation parameters
        max_decoding_length: int,
        eos_token_id: Optional[int],

        clip_visual_model_name: str

    ):
        super().__init__()

        assert policy_lm in SUPPORTED_LMS  # TODO: Support more LMs
        model = policy_lm
        self.device = 0  # TODO
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model,
                                  device=self.device)
        for param in self.generator.model.parameters():
            param.requires_grad = False

        self.logit_bias = logit_bias
        self.fluent = fluent
        self.fluent_top_k = fluent_top_k
        self.max_decoding_length = max_decoding_length
        self.eos_token_id = eos_token_id
        self.clip_visual_model_name = clip_visual_model_name
        model_dim = LM_HIDDEN_SIZES[model]

        self.clip_visual_model = CLIPVisionModel.from_pretrained(clip_visual_model_name).to(self.device)
        for param in self.clip_visual_model.parameters():
            param.requires_grad = False

        self.text_image_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=8, batch_first=True).to(self.device) # check batch_first
        
        # TODO: may add mlp layer after the attention layer for better representation power
        '''self.mlp = _build_one_layer_mlp(in_dim=model_dim,
                                        out_dim=model_dim,
                                        hidden_size=hidden_size).to(self.device)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(_init_weights) '''

    def _attention_forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor: 
        """
        Prompt generation LM is conditioned on image information using Cross Attention:

        CLIP encode the image into image_embed, 
        
        The LM's hidden states as query and image_embed as key and value.
        
        if fluent: compare topk and min value, assign -inf to these logits. What is the use?
        """
        # image: (batch_size, (image_shape)) e.g. (8, 3, 224, 224)
        image = image.to(self.device)
        # image_embed: (batch_size, patch_size, embedding_size) e.g. (8, 50, 768)
        image_embed = self.clip_visual_model(image)['last_hidden_state'] # Last layer's hidden states output
        
        # state: (batch_size, seqence_length, embedding_size) e.g. (8, 1, 768)
        # image_embed as key to select attention of text state
        # attention_output: (batch_size, sequence_length, embedding_size), i.e: (8, 1, 768) same as hidden_states
        attention_output = self.text_image_attention(query=state, 
                                                     key=image_embed, 
                                                     value=image_embed, 
                                                     need_weights=False)[0] # result is tuple: (output)
       
        #mlp_output = self.mlp(state) TODO: may use mlp later
        # logits: (batch_size, 1, embedding_size)
        logits = self.generator.model.lm_head(attention_output)
        #print(f"logits: {logits.shape}")
        if self.fluent:
            lm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(lm_logits, k=self.fluent_top_k)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(lm_logits < min_values,
                                 torch.full_like(logits, float('-inf')),
                                 logits)

        return logits

    # TODO: didn't use _get_generation_cache_all_hidden_states right now
    def teacher_forcing(
        self,
        source_texts: List[str],
        image: torch.Tensor,
        sample_ids: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        state, past_key_values = self._get_generation_cache(source_texts)

        sample_logits = []
        for i in range(sample_ids.shape[-1]):
            logits = self._attention_forward(state, image)
            logits = logits + self.logit_bias

            actions = sample_ids[:, i]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            sample_logits.append(logits.unsqueeze(dim=1))
            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        sample_logits = torch.cat(sample_logits, dim=1)
        output = dict(sample_logits=sample_logits,
                      sample_ids=sample_ids)
        return output

    # TODO: didn't use _get_generation_cache_all_hidden_states right now
    def sample(
        self,
        source_texts: List[str],
        image: torch.Tensor,
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
        eos_token_id: Optional[int],
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        if eos_token_id is not None:
            raise NotImplementedError("Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._attention_forward(state, image)  # [batch_size, vocab_size]
            logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())

            if top_k is not None:
                sampling_logits = _top_k_logits(logits, k=top_k)
            elif top_p is not None:
                sampling_logits = _top_p_logits(logits, p=top_p)
            else:
                sampling_logits = logits

            actions = (torch.distributions.categorical
                       .Categorical(logits=sampling_logits)
                       .sample())  # [batch_size]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))  # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1))
            # [batch_size, 1, vocab_size]

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens for _ in range(sample_ids.shape[0])]).to(self.device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    def greedy_search(self,
                      source_texts: List[str],
                      image: torch.Tensor,
                      max_new_tokens: Optional[int],
                      eos_token_id: Optional[int],
                      **kwargs):
        if eos_token_id is not None:
            raise NotImplementedError("Only support fixed length prompt for now")

        print("greedy_search")
        state, past_key_values = self._get_generation_cache_all_hidden_states(source_texts) #the hidden state of the last token of the Transformer output
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            print("step", i)
            logits = self._attention_forward(state, image)
            logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())
            
            actions = logits[:,-1,:].argmax(dim=-1)  # Take the last token, action: (batch_size)
            #print("action:", actions.shape)
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t]) 
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))
            # keep generating using previous prompt tokens (autoregressive)
            state, past_key_values = self._get_generation_cache_all_hidden_states(token_strs, past_key_values)

        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens for _ in range(sample_ids.shape[0])]).to(self.device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    
    def _get_generation_cache(self,
                              source_texts: List[str],
                              past_key_values=None):
        '''
        input: 
            source_text
            past_key_values(the last key and value parameters in the Transformer)

        Run Transformer and get the output of last token and parameters

        output:
            the hidden state of the last token of the Transformer output
            past_key_values
        '''
        token_encoding = (self.generator.tokenizer(source_texts,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt').to(self.device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = self.generator.model.transformer(input_ids,
                                                   past_key_values=past_key_values,
                                                   use_cache=True)
        last_token_hidden_state = outputs.last_hidden_state[np.arange(input_ids.shape[0]), #batch, last_hidden_state
                                      (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        return last_token_hidden_state, past_key_values
    
    def _get_generation_cache_all_hidden_states(self,
                                source_texts: List[str],
                                past_key_values=None):
        '''
        input: 
            source_text
            past_key_values(the last key and value parameters in the Transformer)

        Run Transformer and get the output of last token and parameters

        output:
            the hidden states of all tokens of the Transformer output
            past_key_values
        '''
        token_encoding = (self.generator.tokenizer(source_texts,
                                        padding=True,
                                        truncation=True,
                                        return_tensors='pt').to(self.device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = self.generator.model.transformer(input_ids,
                                                    past_key_values=past_key_values,
                                                    use_cache=True)
        tokens_hidden_states = outputs.last_hidden_state[np.arange(input_ids.shape[0])]  # (batch, sequence_len, hidden_size)
        past_key_values = outputs.past_key_values
        return tokens_hidden_states, past_key_values

    # Not use batch iamges yet, only one image
    def generate(
        self,
        source_texts: List[str], # beginning as empty prompts with only <EOF>
        image: torch.Tensor,
        do_sample: bool,
        top_k: Optional[int],
        top_p: float,
        #num_beams: int,
        max_new_tokens: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        
        if max_new_tokens is None:
            max_new_tokens = self.max_decoding_length
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        is_greedy_gen_mode = (do_sample == False) 
        is_sample_gen_mode = (do_sample == True) 
        assert is_greedy_gen_mode or is_sample_gen_mode

        if is_greedy_gen_mode:
            return self.greedy_search(source_texts=source_texts,
                                      image=image,
                                      max_new_tokens=max_new_tokens,
                                      eos_token_id=eos_token_id)
        elif is_sample_gen_mode:
            return self.sample(source_texts=source_texts,
                               image=image,
                               top_k=top_k,
                               top_p=top_p,
                               max_new_tokens=max_new_tokens,
                               eos_token_id=eos_token_id)


