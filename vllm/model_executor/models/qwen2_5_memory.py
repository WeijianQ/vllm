from typing import Optional, Tuple, TypedDict, Mapping
import torch
import torch.nn as nn
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from typing import Iterable, Optional, Set, Tuple, Union
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.interfaces import SupportsMultiModal, MultiModalEmbeddings
from vllm.model_executor.sampling_metadata import SamplingMetadata

from vllm.model_executor.models.utils import merge_multimodal_embeddings, maybe_prefix, init_vllm_registered_model, WeightsMapper


class MemoryInput(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

from transformers import AutoConfig, AutoModelForCausalLM
class Qwen2_5_MemoryEncoder(nn.Module):
    """
    Minimal HF-based memory encoder:
      - __init__: record cfg, build a linear head
      - forward: HF prefill -> last hidden -> masked mean -> linear
      - load_weights: load only 'embed_head.*' from iterable
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.hf_config = vllm_config.model_config.hf_config
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        cfg.output_hidden_states = True

        device = self.vllm_config.device_config.device   # torch.device("cuda")
        dtype  = getattr(self.vllm_config, "dtype", torch.float16)

        # Use from_config instead of from_pretrained - no weight loading!
        self.hf_model = AutoModelForCausalLM.from_config(
            cfg,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device).eval()

        del self.hf_model.lm_head

    @torch.no_grad()
    def encode(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, position_ids: torch.LongTensor=None) -> torch.FloatTensor:
        return self.hf_model.encode(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        state_dict = {name: data for name, data in weights}
        # Remove lm_head since we deleted it in __init__
        if "lm_head.weight" in state_dict:
            del state_dict["lm_head.weight"]
        self.hf_model.load_state_dict(state_dict, strict=False)
        return set(state_dict.keys())
        
from vllm.multimodal.processing import BaseProcessingInfo, BaseMultiModalProcessor, PromptReplacement
class Qwen2_5_MemoryProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"memory": 100,}

#### DUMMY INPUTS BUILDER ####
from vllm.multimodal.profiling import BaseDummyInputsBuilder
class Qwen2_5_MemoryDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2_5_MemoryProcessingInfo]):
    def get_dummy_text(self, mm_counts) -> str:
        num_memories = mm_counts.get("memory", 0)
        memory_token = self.info.get_hf_config().memory_pad_token
        return f"{memory_token} dummy text " * num_memories

    def get_dummy_mm_data(self, seq_len, mm_counts):
        num_memories = mm_counts.get("memory", 0)
        return {
            "memory": ["This is a dummy memory" for _ in range(num_memories)]
        }

##### 
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
class Qwen2_5_MemoryMultiModalProcessor(BaseMultiModalProcessor[Qwen2_5_MemoryProcessingInfo]):

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return dict(
            memory_input_ids=MultiModalFieldConfig.batched("memory"),
            memory_attention_mask=MultiModalFieldConfig.batched("memory"),
        )

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        modality = "memory"
        num_items = len(mm_items.get(modality, []))
        if num_items == 0:
            return []

        tokenizer = self.info.get_tokenizer()
        mem_pad_id = tokenizer.convert_tokens_to_ids('<|mem_pad|>')
        return [
                PromptReplacement(
                    modality=modality,
                    target=[mem_pad_id],
                    replacement=PromptUpdateDetails.select_token_id(
                        seq=[mem_pad_id],         # the full text we put back into the prompt
                        embed_token_id=mem_pad_id # only this ID is considered a feature placeholder
                    ),
                )
            ]

from vllm.multimodal import MULTIMODAL_REGISTRY

@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_MemoryMultiModalProcessor,
    info=Qwen2_5_MemoryProcessingInfo,
    dummy_inputs=Qwen2_5_MemoryDummyInputsBuilder,
)
class Qwen2_5_MemoryForCausalLM(nn.Module, SupportsMultiModal):
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )
        self.encoder = Qwen2_5_MemoryEncoder(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "encoder"))
        self.language_model_embed_dtype = next(self.language_model.model.embed_tokens.parameters()).dtype

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        # print(input_ids.shape, len(multimodal_embeddings), multimodal_embeddings[0].shape)
        # if input_ids.shape[-1] > 100:
            # from src.utils import wait_for_debugger
            # wait_for_debugger()
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.memory_pad_token_id])
        return inputs_embeds

    def _pad_mem_ids(self, ids_or_mask: torch.Tensor, target_len: int, pad_id: int) -> torch.Tensor:
        """Pad a single tensor to target length with left padding."""
        batch_size, seq_len = ids_or_mask.shape
        if seq_len >= target_len:
            return ids_or_mask[:, -target_len:]
        padded_ = torch.full((batch_size, target_len), pad_id, dtype=ids_or_mask.dtype, device=ids_or_mask.device)
        padded_[:, -seq_len:] = ids_or_mask
        # we do left padding
        return padded_

    def _collate_memory_inputs(
        self,
        mem_ids_list: list[torch.Tensor],
        mem_mask_list: Optional[list[torch.Tensor]]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], list[int]]:
        """
        Collate (pad and batch) a list of memory input tensors.

        Args:
            mem_ids_list: List of memory input_ids tensors, each of shape (num_items, seq_len)
            mem_mask_list: Optional list of attention masks

        Returns:
            Tuple of:
                - Batched and padded mem_ids: (total_items, max_len)
                - Batched and padded mem_mask: (total_items, max_len) or None
                - num_sequences: List of number of items per request
        """
        max_len = max(m.shape[1] for m in mem_ids_list)
        num_sequences = [m.shape[0] for m in mem_ids_list]

        # Pad all tensors to max_len
        mem_ids_padded = [self._pad_mem_ids(m, max_len, self.config.memory_pad_token_id) for m in mem_ids_list]
        mem_ids = torch.concat(mem_ids_padded, dim=0)  # (total_batch, max_len)

        # Handle mask
        if mem_mask_list is not None:
            mem_mask_padded = [self._pad_mem_ids(m, max_len, 0) for m in mem_mask_list]
            mem_mask = torch.concat(mem_mask_padded, dim=0)
        else:
            mem_mask = None

        return mem_ids, mem_mask, num_sequences

    def _encode_memory_batch(
        self,
        mem_ids: torch.Tensor,
        mem_mask: Optional[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Encode a batch of memory inputs.

        Args:
            mem_ids: Memory input ids tensor of shape (batch_size, seq_len)
            mem_mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            List of encoded embeddings, one per item in the batch
        """
        # Ensure tensors are on correct device and dtype
        mem_ids = mem_ids.to(dtype=torch.long, device=self.encoder.hf_model.device)

        if mem_mask is None:
            mem_mask = torch.ones_like(mem_ids, dtype=torch.long)
        else:
            mem_mask = mem_mask.to(dtype=torch.long, device=self.encoder.hf_model.device)

        # Encode
        stacked_embeddings = self.encoder.encode(input_ids=mem_ids, attention_mask=mem_mask)

        # Convert to list of individual embeddings with correct dtype
        dispatched_embeddings = [
            embedding.unsqueeze(0).to(dtype=self.language_model_embed_dtype)
            for embedding in stacked_embeddings
        ]

        return dispatched_embeddings

    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        mem_ids = kwargs.pop("memory_input_ids", None) # batch_size, 1 , max_len
        mem_mask = kwargs.pop("memory_attention_mask", None)   # (B, L)

        if mem_ids is None:
            return None

        # if isinstance(mem_ids, list) and len(mem_ids) > 0:
        #     print(f"DEBUG: mem_ids is a list, len: {len(mem_ids)}")

        # Case 1: Single tensor with shape (batch, 1, seq_len)
        if isinstance(mem_ids, torch.Tensor) and mem_ids.shape[1] == 1:
            mem_ids = mem_ids.squeeze(1).squeeze(1)  # (batch, seq_len)
            mem_mask = mem_mask.squeeze(1) if mem_mask is not None else None
            return self._encode_memory_batch(mem_ids, mem_mask)

        # Case 2: List of tensors (need padding and batching)
        else:
            # Count total items
            total_items = sum(m.shape[0] for m in mem_ids)
            num_requests = len(mem_ids)
            # print(f"DEBUG: Total {total_items} items from {num_requests} requests")

            # Flatten: convert list of (num_items_per_request, seq_len) to flat list
            flat_mem_ids = []
            flat_mem_masks = []
            request_boundaries = [0]  # Track which items belong to which request

            for i, mem_id_tensor in enumerate(mem_ids):
                num_items = mem_id_tensor.shape[0]
                for j in range(num_items):
                    flat_mem_ids.append(mem_id_tensor[j:j+1])  # Keep as (1, seq_len)
                    if mem_mask is not None:
                        flat_mem_masks.append(mem_mask[i][j:j+1])
                request_boundaries.append(request_boundaries[-1] + num_items)

            # Process in batches of 16
            batch_size = 16
            num_batches = (total_items + batch_size - 1) // batch_size
            # print(f"  Processing {total_items} items in {num_batches} batches (batch_size={batch_size})")

            all_embeddings = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_items)
                batch_mem_ids = flat_mem_ids[start_idx:end_idx]
                batch_mem_masks = flat_mem_masks[start_idx:end_idx] if mem_mask is not None else None

                # print(f"    Batch {batch_idx + 1}/{num_batches}: items [{start_idx}:{end_idx}] (size={end_idx - start_idx})")

                # Collate this batch
                collated_ids, collated_mask, _ = self._collate_memory_inputs(batch_mem_ids, batch_mem_masks)
                # print(f"      Collated shape: {collated_ids.shape}")

                # Encode this batch
                batch_embeddings = self._encode_memory_batch(collated_ids, collated_mask)
                all_embeddings.extend(batch_embeddings)

            # print(f"  Total embeddings collected: {len(all_embeddings)}")

            # Group embeddings by request
            dispatched_embeddings = []
            for i in range(num_requests):
                start_idx = request_boundaries[i]
                end_idx = request_boundaries[i + 1]
                request_embeddings = all_embeddings[start_idx:end_idx]
                # Stack into single tensor
                stacked = torch.cat(request_embeddings, dim=0)
                dispatched_embeddings.append(stacked.to(dtype=self.language_model_embed_dtype))

            return dispatched_embeddings


    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        weights = list(weights)
        loaded = set()
        weights_wo_embed_head = ((name, data) for name, data in weights if not name.startswith("embed_head."))
        language_model_loaded = self.language_model.load_weights(weights_wo_embed_head)
        loaded.update([f"language_model.{name}" for name in language_model_loaded])
        encoder_loaded = self.encoder.load_weights(weights)
        loaded.update([f"encoder.hf_model.{name}" for name in encoder_loaded])
        return loaded

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: "SamplingMetadata",
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            # TODO dont know if this is correct
            inputs_embeds = None
        elif inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(
                input_ids,
                # probably we can add some options here, but we let vllm to do the merge
            )
            input_ids = None
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states
