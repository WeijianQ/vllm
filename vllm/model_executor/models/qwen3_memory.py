from typing import Optional, Tuple, TypedDict, Mapping, Literal
import torch
import torch.nn as nn
from typing import Iterable, Optional, Set, Tuple, Union
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsMultiModal, MultiModalEmbeddings
from vllm.model_executor.sampling_metadata import SamplingMetadata

from vllm.model_executor.models.utils import merge_multimodal_embeddings, maybe_prefix, init_vllm_registered_model


class Qwen3MemoryEmbeddingInputs(TypedDict):
    type: Literal["memory_embeds"]
    memory_embeds: torch.Tensor
    """
    Memory embeddings tensor.

    Tensor shape: `(num_memory_features, hidden_size)`
    - `num_memory_features` varies based on the number of memory items.
    - `hidden_size` must match the hidden size of language model backbone.
    """


from vllm.multimodal.processing import BaseProcessingInfo, BaseMultiModalProcessor, PromptReplacement
class Qwen3MemoryProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"memory": 100,}


#### DUMMY INPUTS BUILDER ####
from vllm.multimodal.profiling import BaseDummyInputsBuilder
class Qwen3MemoryDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3MemoryProcessingInfo]):
    def get_dummy_text(self, mm_counts) -> str:
        num_memories = mm_counts.get("memory", 0)
        memory_token = self.info.get_hf_config().memory_pad_token
        return f"{memory_token} dummy text " * num_memories

    def get_dummy_mm_data(self, seq_len, mm_counts):
        num_memories = mm_counts.get("memory", 0)
        hf_config = self.info.get_hf_config()
        hidden_size = hf_config.hidden_size
        # Return dummy tensor embeddings as a single 3D tensor (batch, num_tokens, hidden_size)
        # This prevents extra dimension being added during batching
        # For profiling, we assume 1 token per memory item
        return {
            "memory": torch.randn(num_memories, 1, hidden_size)
        }


#####
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
class Qwen3MemoryMultiModalProcessor(BaseMultiModalProcessor[Qwen3MemoryProcessingInfo]):

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return dict(
            memory_embeds=MultiModalFieldConfig.batched("memory"),
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
    Qwen3MemoryMultiModalProcessor,
    info=Qwen3MemoryProcessingInfo,
    dummy_inputs=Qwen3MemoryDummyInputsBuilder,
)
class Qwen3MemoryForCausalLM(nn.Module, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )
        self.language_model_embed_dtype = next(self.language_model.model.embed_tokens.parameters()).dtype

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.config.memory_pad_token_id])
        return inputs_embeds

    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        memory_embeds = kwargs.pop("memory_embeds", None)

        if memory_embeds is None:
            return None

        # Only accept pre-computed embeddings
        if not isinstance(memory_embeds, (torch.Tensor, list)):
            raise ValueError(
                f"memory_embeds must be torch.Tensor or list of tensors, "
                f"got {type(memory_embeds)}. "
                f"This model only accepts pre-computed memory embeddings."
            )

        # Convert to list of 2D tensors
        if isinstance(memory_embeds, torch.Tensor):
            if memory_embeds.ndim == 2:
                # (num_items, hidden_size) -> list of (1, hidden_size)
                memory_embeds_list = [memory_embeds[i:i+1] for i in range(memory_embeds.shape[0])]
            elif memory_embeds.ndim == 3:
                # (batch, num_tokens, hidden_size) -> list of (num_tokens, hidden_size)
                memory_embeds_list = [memory_embeds[i] for i in range(memory_embeds.shape[0])]
            elif memory_embeds.ndim == 4:
                # (batch, inner_batch, num_tokens, hidden_size) -> list of (num_tokens, hidden_size)
                # This happens during profiling when dummy 3D tensors are batched together
                # Reshape to (batch * inner_batch, num_tokens, hidden_size) then split
                batch_size = memory_embeds.shape[0] * memory_embeds.shape[1]
                reshaped = memory_embeds.reshape(batch_size, memory_embeds.shape[2], memory_embeds.shape[3])
                memory_embeds_list = [reshaped[i] for i in range(reshaped.shape[0])]
            else:
                raise ValueError(f"Unexpected memory_embeds tensor shape: {memory_embeds.shape}")
        else:
            # Already a list - ensure each element is 2D
            memory_embeds_list = []
            for emb in memory_embeds:
                if emb.ndim == 2:
                    memory_embeds_list.append(emb)
                elif emb.ndim == 3:
                    # Squeeze batch dimension if it's 1
                    if emb.shape[0] == 1:
                        memory_embeds_list.append(emb.squeeze(0))
                    else:
                        raise ValueError(f"Unexpected embedding shape in list: {emb.shape}")
                else:
                    raise ValueError(f"Unexpected embedding ndim in list: {emb.ndim}, shape: {emb.shape}")

        # Convert to correct dtype and return
        dispatched_embeddings = [
            emb.to(dtype=self.language_model_embed_dtype)
            for emb in memory_embeds_list
        ]

        return dispatched_embeddings

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        weights = list(weights)
        loaded = set()

        # Load language model weights
        # Filter out memory-specific modules that are not used in vLLM inference
        # (memory embeddings are pre-computed externally)
        skip_prefixes = ("memory_projector", "memory_qformer")
        weights = [(name, weight) for name, weight in weights
                   if not any(name.startswith(prefix) for prefix in skip_prefixes)]
        language_model_loaded = self.language_model.load_weights(weights)
        loaded.update([f"language_model.{name}" for name in language_model_loaded])

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
            inputs_embeds = None
        elif inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(
                input_ids,
            )
            input_ids = None
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states
