import copy
import importlib.util

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
try:
    from unsloth import FastLanguageModel  # type: ignore
except ImportError:  # pragma: no cover - optional in stable non-Unsloth runs
    FastLanguageModel = None

from typing import Any, Dict, List, Optional, Sequence, Union
from pathlib import Path

from bioreason2.models.pl.processing_pl import PLProcessor
from bioreason2.models.pl.chat_template_pl import get_chat_template
from bioreason2.models.protein_encoder import create_protein_encoder
from bioreason2.models.go_graph_encoder import create_go_graph_encoder_pipeline
from bioreason2.models.special_tokens import get_all_special_tokens, get_token


def _load_text_tokenizer(model_name: str, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name, **kwargs)


def _normalize_attn_implementation(value: Optional[str]) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return "auto"
    if normalized in {"flash", "fa2", "flash-attention-2", "flashattention2"}:
        return "flash_attention_2"
    return normalized


def _flash_attn_is_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _attention_candidates(requested: Optional[str]) -> List[str]:
    normalized = _normalize_attn_implementation(requested)
    if normalized == "auto":
        candidates: List[str] = []
        if torch.cuda.is_available() and _flash_attn_is_available():
            candidates.append("flash_attention_2")
        candidates.extend(["sdpa", "eager"])
        return candidates
    if normalized == "flash_attention_2" and not _flash_attn_is_available():
        return ["sdpa", "eager"]
    if normalized == "sdpa":
        return ["sdpa", "eager"]
    if normalized == "eager":
        return ["eager"]
    return [normalized]


def _is_attention_backend_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "flashattention",
            "flash attention",
            "flash_attn",
            "attn_implementation",
            "attention implementation",
            "scaled dot product attention",
            "sdpa",
        )
    )


def _load_text_model_with_attention_fallback(model_name: str, **kwargs):
    requested = kwargs.get("attn_implementation")
    candidates = _attention_candidates(requested)
    last_error: Optional[BaseException] = None
    for index, candidate in enumerate(candidates):
        load_kwargs = dict(kwargs)
        load_kwargs["attn_implementation"] = candidate
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            if requested not in (None, "", candidate):
                print(
                    f"Warning: requested attention backend '{requested}' resolved to '{candidate}' "
                    f"for {model_name}."
                )
            return model, candidate
        except (ImportError, RuntimeError, ValueError) as exc:
            last_error = exc
            if index + 1 >= len(candidates) or not _is_attention_backend_error(exc):
                raise
            next_candidate = candidates[index + 1]
            print(
                f"Warning: attention backend '{candidate}' is unavailable for {model_name}: "
                f"{str(exc).splitlines()[0]} Falling back to '{next_candidate}'."
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load text model {model_name} with any attention backend.")


def _get_target_modules(model):
    """Get target modules for LoRA fine-tuning."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class ProteinLLMModel(nn.Module):
    """
    A combined model that processes both protein sequences and text inputs.

    The model uses a protein encoder (ESM3 or ESM-C) to extract features from protein sequences
    and a text model (LLM) to process text inputs and generate responses. The protein features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        text_model_name: str,
        protein_model_name: str = "esm3_sm_open_v1",
        cache_dir: Optional[str] = None,
        max_length_protein: int = 2048,
        max_length_text: int = 4096,
        text_model_finetune: bool = True,
        protein_model_finetune: bool = False,
        protein_train_layer_start: int = 36,
        protein_embedding_layer: int = -1,
        go_model_finetune: bool = True,
        attn_implementation: str = "auto",
        go_obo_path: Optional[str] = None,
        precomputed_embeddings_path: Optional[str] = None,
        go_hidden_dim: int = 512,
        go_num_gat_layers: int = 3,
        go_num_heads: int = 8,
        go_num_reduced_embeddings: int = 200,  # Update processing_pl.py to use this as well
        go_embedding_dim: int = 2560,
        quantization_config: Optional[object] = None,  # QLoRA quantization config
        load_in_4bit: bool = False,
        unified_go_encoder: bool = False,
        use_unsloth: bool = True,
    ):
        """
        Initialize the ProteinLLMModel.

        Args:
            text_model_name: Name of the text model to be used.
            protein_model_name: Name of the protein model to be used (ESM3 or ESM-C).
            cache_dir: Directory to cache the models.
            max_length_protein: Maximum length of protein sequences. Defaults to 2048.
            max_length_text: Maximum length of text sequences. Defaults to 4096.
            text_model_finetune: Whether to finetune the text model. Defaults to True.
            protein_model_finetune: Whether to finetune the protein model. Defaults to False.
            protein_train_layer_start: ESM3 layer to start training from. Use -1 or >=total_blocks for output heads only, 0 for all transformer layers. Defaults to 36.
            protein_embedding_layer: ESM3 layer to extract embeddings from. Use -1 for final output (default), 0-N for specific transformer layers. Only works with ESM3 models.
            go_model_finetune: Whether to finetune the GO graph encoder. Defaults to True.
            attn_implementation: Attention implementation to use. Defaults to "auto",
                which prefers FlashAttention2 and falls back to SDPA/Eager when needed.
            go_obo_path: Path to GO ontology OBO file. If None, GO encoder will be disabled.
            precomputed_embeddings_path: Directory with GO embeddings .safetensors files.
            go_hidden_dim: Hidden dimension for GO GAT layers. Defaults to 512.
            go_num_gat_layers: Number of GAT layers in GO encoder. Defaults to 3.
            go_num_heads: Number of attention heads in GO GAT. Defaults to 8.
            go_num_reduced_embeddings: Number of reduced embeddings per GO namespace. Defaults to 200.
            go_embedding_dim: GO embedding dimension. Defaults to 2560.
            quantization_config: QLoRA quantization configuration for 4-bit training. Defaults to None.
            load_in_4bit: Whether to load the model in 4-bit for unsloth. Defaults to False.
            unified_go_encoder: If True, use unified GOGraphEncoderUnified; if False, use original GOGraphEncoder
            use_unsloth: If True, use Unsloth for faster training. Defaults to True.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.protein_model_finetune = protein_model_finetune
        self.protein_train_layer_start = protein_train_layer_start
        self.protein_embedding_layer = protein_embedding_layer
        self.go_model_finetune = go_model_finetune
        self.max_length_protein = max_length_protein
        self.max_length_text = max_length_text
        self.unified_go_encoder = unified_go_encoder
        self.use_unsloth = use_unsloth
        self.attn_implementation = _normalize_attn_implementation(attn_implementation)

        if use_unsloth:
            if FastLanguageModel is None:
                raise RuntimeError(
                    "use_unsloth=True but the unsloth package is not installed. "
                    "Install unsloth or initialize ProteinLLMModel with use_unsloth=False."
                )
            self.text_model, self.text_tokenizer = FastLanguageModel.from_pretrained(
                model_name=text_model_name,
                max_seq_length=max_length_text + max_length_protein + go_num_reduced_embeddings + 8,    # Use 8 for special tokens
                dtype=torch.bfloat16,
                load_in_4bit=load_in_4bit,
                cache_dir=cache_dir,
                trust_remote_code=True,
                device_map={"": "cpu"},
            )
        else:
            text_model_kwargs = {
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "attn_implementation": attn_implementation,
            }
            if quantization_config is not None:
                text_model_kwargs["quantization_config"] = quantization_config

            self.text_model, self.attn_implementation = _load_text_model_with_attention_fallback(
                text_model_name,
                **text_model_kwargs,
            )
            self.text_tokenizer = _load_text_tokenizer(
                text_model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

        self.text_config = self.text_model.config
        self.text_tokenizer.chat_template = get_chat_template(text_model_name)

        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Add special tokens from centralized module
        all_special_tokens = get_all_special_tokens()
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens})
        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("protein_pad"))
        self.go_token_id = self.text_tokenizer.convert_tokens_to_ids(get_token("go_graph_pad"))

        self.text_model.resize_token_embeddings(len(self.text_tokenizer))

        # Load the protein encoder (ESM3 or ESM-C). When the text checkpoint is a
        # materialized BioReason-Pro artifact, prefer its bundled protein_model/.
        resolved_protein_model_name = protein_model_name
        text_model_path = Path(text_model_name).expanduser()
        checkpoint_protein_model = text_model_path / "protein_model"
        if text_model_path.exists() and checkpoint_protein_model.is_dir():
            resolved_protein_model_name = str(checkpoint_protein_model)
            print(f"📁 Using checkpoint-bundled protein model from {checkpoint_protein_model}")

        self.protein_encoder = create_protein_encoder(
            resolved_protein_model_name,
            inference_mode=not protein_model_finetune,
            embedding_layer=protein_embedding_layer
        )
        self.protein_model = self.protein_encoder.model

        # Get embedding dimensions
        self.text_hidden_size = self.text_config.hidden_size
        self.protein_hidden_size = self.protein_encoder.embedding_dim

        # Initialize GO graph encoder if paths are provided
        self.go_encoder = None
        self.go_embeddings_cache = {}
        if go_obo_path is not None and precomputed_embeddings_path is not None:
            self.go_encoder = create_go_graph_encoder_pipeline(
                go_obo_path=go_obo_path,
                precomputed_embeddings_path=precomputed_embeddings_path,
                hidden_dim=go_hidden_dim,
                num_gat_layers=go_num_gat_layers,
                num_heads=go_num_heads,
                num_reduced_embeddings=go_num_reduced_embeddings,
                embedding_dim=go_embedding_dim,
                unified_go_encoder=unified_go_encoder
            )
        # Always create projection layer for GO embeddings so checkpoint-bundled
        # GO embeddings can be used even when the GO encoder source directory is absent.
        self.go_projection = nn.Sequential(
            nn.Linear(go_embedding_dim, self.text_hidden_size),
            nn.GELU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
        )

        # Create projection layer to map protein embeddings to text model's embedding space
        self.protein_projection = nn.Sequential(
            nn.Linear(self.protein_hidden_size, self.text_hidden_size),
            nn.GELU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
        )

        # Initialize all models in eval mode with frozen parameters by default
        # Training setup will be handled by train_protein_llm.py
        self._setup_default_eval_mode()

        # Create processor for handling inputs
        self.processor = PLProcessor(tokenizer=self.text_tokenizer)

    def _setup_default_eval_mode(self):
        """
        Set all model components to eval mode with frozen parameters by default.
        Training setup will be handled by train_protein_llm.py.
        """
        # Text model: eval mode, frozen
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Protein encoder: use proper API to set inference mode
        self.protein_encoder.set_inference_mode(inference_mode=not self.protein_model_finetune)
        
        # Protein projection: eval mode, frozen
        self.protein_projection.eval()
        for param in self.protein_projection.parameters():
            param.requires_grad = False
            
        # GO encoder: eval mode, frozen
        if self.go_encoder is not None:
            self.go_encoder.eval()
            for param in self.go_encoder.parameters():
                param.requires_grad = False
        
        # GO projection: eval mode, frozen
        if self.go_projection is not None:
            self.go_projection.eval()
            for param in self.go_projection.parameters():
                param.requires_grad = False

    def process_protein_embeddings(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Process protein sequences and structures to obtain embeddings using the protein encoder.

        Args:
            protein_sequences: List of protein sequence strings
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch
            structure_coords: Optional tensor containing the structure coordinates

        Returns:
            List of tensor embeddings for each batch item
        """
        # Use the protein encoder to get embeddings
        batch_protein_embeddings = self.protein_encoder.encode_sequences(
            protein_sequences=protein_sequences,
            batch_idx_map=batch_idx_map,
            batch_size=batch_size,
            structure_coords=structure_coords,
        )

        # Project all embeddings to text embedding space
        for i in range(batch_size):
            if batch_protein_embeddings[i].numel() > 0:  # Check if tensor is not empty
                batch_protein_embeddings[i] = batch_protein_embeddings[i].to(
                    device=self.protein_projection[0].weight.device,
                    dtype=self.protein_projection[0].weight.dtype,
                )
                batch_protein_embeddings[i] = self.protein_projection(batch_protein_embeddings[i])
                batch_protein_embeddings[i] = self._sanitize_embedding_tensor(batch_protein_embeddings[i])
            else:
                # Ensure empty tensors have correct dimensions
                batch_protein_embeddings[i] = torch.zeros(
                    (0, self.text_hidden_size),
                    device=self.protein_projection[0].weight.device,
                    dtype=self.protein_projection[0].weight.dtype,
                )

        return batch_protein_embeddings

    def process_go_aspects(
        self,
        go_aspects: Optional[List[str]] = None,
        batch_size: int = 1,
    ) -> Optional[List[torch.Tensor]]:
        """
        Process GO aspects to obtain embeddings using the GO graph encoder.
        Each example gets its own aspect-specific embeddings or all aspects combined.

        Args:
            go_aspects: List of GO aspect strings for each batch item. If None or
                       individual items are None, defaults to "all" aspect.
            batch_size: Number of items in the batch

        Returns:
            Optional list of tensors with GO embeddings, one per batch item.
            Each tensor has shape (200, text_hidden_size) for specific aspect
            or combined all aspects when aspect is None or "all".
            Returns None if no GO encoder is available.
        """
        if go_aspects is None:
            return None
        if self.go_encoder is None and not self.go_embeddings_cache:
            return None

        batch_go_embeddings = []

        if self.unified_go_encoder:
            # Namespace doesn't matter for unified encoder. Prefer a checkpoint-bundled
            # cached tensor when available, otherwise compute via the encoder.
            if "all" in self.go_embeddings_cache:
                reduced_embeddings = self.go_embeddings_cache["all"]
            else:
                reduced_embeddings = self.go_encoder("all")  # (200, 2560)
                if not self.go_model_finetune:
                    self.go_embeddings_cache["all"] = reduced_embeddings

            # Project to text embedding space
            if self.go_projection is not None:
                reduced_embeddings = reduced_embeddings.to(
                    device=self.go_projection[0].weight.device,
                    dtype=self.go_projection[0].weight.dtype,
                )
                reduced_embeddings = self.go_projection(reduced_embeddings)  # (200, text_hidden_size)
                reduced_embeddings = self._sanitize_embedding_tensor(reduced_embeddings)

            # Duplicate for all batch items
            for i in range(batch_size):
                batch_go_embeddings.append(reduced_embeddings)
        else:
            # Process each example's aspect separately for non-unified encoder
            for i in range(batch_size):
                # Use default "all" aspect if no specific aspect is provided
                if i < len(go_aspects) and go_aspects[i] is not None:
                    aspect = go_aspects[i]
                else:
                    aspect = "all"

                if aspect in self.go_embeddings_cache:
                    reduced_embeddings = self.go_embeddings_cache[aspect]
                elif "all" in self.go_embeddings_cache:
                    reduced_embeddings = self.go_embeddings_cache["all"]
                else:
                    # Get reduced embeddings for this specific aspect (200, 2560)
                    reduced_embeddings = self.go_encoder(aspect)
                    if not self.go_model_finetune:
                        self.go_embeddings_cache[aspect] = reduced_embeddings

                # Project to text embedding space
                if self.go_projection is not None:
                    reduced_embeddings = reduced_embeddings.to(
                        device=self.go_projection[0].weight.device,
                        dtype=self.go_projection[0].weight.dtype,
                    )
                    reduced_embeddings = self.go_projection(reduced_embeddings)  # (200, text_hidden_size)
                    reduced_embeddings = self._sanitize_embedding_tensor(reduced_embeddings)
                batch_go_embeddings.append(reduced_embeddings)

        return batch_go_embeddings

    def load_precomputed_go_embedding_cache(self, embedding_path: str, aspect: str = "all") -> None:
        """
        Load checkpoint-bundled GO embeddings when the full GO encoder source
        directory is not available locally.
        """
        cache_path = Path(embedding_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"GO embedding cache not found: {cache_path}")

        cached_embedding = torch.load(cache_path, map_location="cpu")
        if not isinstance(cached_embedding, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor in {cache_path}, got {type(cached_embedding)!r}")

        self.go_embeddings_cache[aspect] = cached_embedding
        print(f"✅ Loaded checkpoint-bundled GO embedding cache from {cache_path} for aspect '{aspect}'")

    @staticmethod
    def _sanitize_embedding_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Clamp NaN/Inf values before they reach text generation."""
        return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)

    def _truncate_protein_inputs(
        self,
        protein_sequences: Optional[List[str]],
        structure_coords: Optional[torch.Tensor],
    ) -> tuple[Optional[List[str]], Optional[torch.Tensor]]:
        if protein_sequences is not None:
            protein_sequences = [str(seq)[: self.max_length_protein] for seq in protein_sequences]
        if isinstance(structure_coords, torch.Tensor) and structure_coords.shape[1] > self.max_length_protein:
            structure_coords = structure_coords[:, : self.max_length_protein, ...]
        return protein_sequences, structure_coords

    def build_multimodal_cache(
        self,
        *,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        batch_size: int,
        structure_coords: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Precompute projected protein / GO features for reuse across rollouts."""
        cache: Dict[str, Any] = {
            "batch_size": int(batch_size),
            "protein_embeddings": None,
            "go_embeddings": None,
        }
        protein_sequences, structure_coords = self._truncate_protein_inputs(protein_sequences, structure_coords)
        if protein_sequences is not None and batch_idx_map is not None:
            cache["protein_embeddings"] = self.process_protein_embeddings(
                protein_sequences,
                batch_idx_map,
                batch_size,
                structure_coords=structure_coords,
            )
        if go_aspects is not None:
            cache["go_embeddings"] = self.process_go_aspects(go_aspects, batch_size)
        return cache

    def expand_multimodal_cache(self, cache: Optional[Dict[str, Any]], repeat_count: int) -> Optional[Dict[str, Any]]:
        """Repeat a single-example cache across rollout copies without re-encoding proteins."""
        if cache is None:
            return None
        if repeat_count <= 0:
            raise ValueError(f"repeat_count must be positive, got {repeat_count}")

        protein_embeddings = cache.get("protein_embeddings")
        go_embeddings = cache.get("go_embeddings")
        expanded_cache: Dict[str, Any] = {
            "batch_size": repeat_count,
            "protein_embeddings": None,
            "go_embeddings": None,
        }
        if protein_embeddings:
            if len(protein_embeddings) != 1:
                raise ValueError(
                    "expand_multimodal_cache currently expects a single-example cache, "
                    f"got {len(protein_embeddings)} protein embedding groups"
                )
            expanded_cache["protein_embeddings"] = [protein_embeddings[0] for _ in range(repeat_count)]
        if go_embeddings:
            if len(go_embeddings) != 1:
                raise ValueError(
                    "expand_multimodal_cache currently expects a single-example cache, "
                    f"got {len(go_embeddings)} GO embedding groups"
                )
            expanded_cache["go_embeddings"] = [go_embeddings[0] for _ in range(repeat_count)]
        return expanded_cache

    def _apply_multimodal_replacements(
        self,
        *,
        input_ids: torch.Tensor,
        text_inputs_embeds: torch.Tensor,
        multimodal_cache: Optional[Dict[str, Any]] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
    ) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])

        if multimodal_cache is None:
            multimodal_cache = self.build_multimodal_cache(
                protein_sequences=protein_sequences,
                batch_idx_map=batch_idx_map,
                batch_size=batch_size,
                structure_coords=structure_coords,
                go_aspects=go_aspects,
            )

        batch_protein_embeds = multimodal_cache.get("protein_embeddings")
        if batch_protein_embeds:
            mask = input_ids == self.protein_token_id
            n_protein_tokens = mask.sum().item()
            protein_embeds_flat = torch.cat(batch_protein_embeds, dim=0)
            n_protein_features = protein_embeds_flat.shape[0]

            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: "
                    f"features {n_protein_features}, tokens: {n_protein_tokens}"
                )

            protein_embeds_flat = protein_embeds_flat.to(dtype=text_inputs_embeds.dtype)
            if n_protein_tokens > 0:
                orig_shape = text_inputs_embeds.shape
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                diff = protein_embeds_flat - embeds_2d.index_select(0, idx)
                embeds_2d = embeds_2d.scatter_add(0, idx.unsqueeze(1).expand(-1, hidden_size), diff)
                text_inputs_embeds = embeds_2d.view(orig_shape)
                text_inputs_embeds = self._sanitize_embedding_tensor(text_inputs_embeds)

        go_embeddings = multimodal_cache.get("go_embeddings")
        if go_embeddings:
            go_mask = input_ids == self.go_token_id
            n_go_tokens = go_mask.sum().item()
            go_embeds_flat = torch.cat([emb for emb in go_embeddings if emb.numel() > 0], dim=0)
            n_go_features = go_embeds_flat.shape[0] if go_embeds_flat.numel() > 0 else 0

            if n_go_features != n_go_tokens:
                raise ValueError(
                    f"GO embeddings and GO tokens do not match: embeddings {n_go_features}, tokens: {n_go_tokens}"
                )

            if n_go_tokens > 0:
                go_embeds_flat = go_embeds_flat.to(dtype=text_inputs_embeds.dtype)
                orig_shape = text_inputs_embeds.shape
                hidden_size = orig_shape[-1]
                embeds_2d = text_inputs_embeds.view(-1, hidden_size)
                mask_flat = go_mask.view(-1)
                idx = mask_flat.nonzero(as_tuple=False).squeeze(1)
                diff = go_embeds_flat - embeds_2d.index_select(0, idx)
                embeds_2d = embeds_2d.scatter_add(0, idx.unsqueeze(1).expand(-1, hidden_size), diff)
                text_inputs_embeds = embeds_2d.view(orig_shape)
                text_inputs_embeds = self._sanitize_embedding_tensor(text_inputs_embeds)

        return text_inputs_embeds

    def prepare_multimodal_prefix(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        multimodal_cache: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs_embeds once so caller can reuse the multimodal prefix."""
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        text_inputs_embeds = self._apply_multimodal_replacements(
            input_ids=input_ids,
            text_inputs_embeds=text_inputs_embeds,
            multimodal_cache=multimodal_cache,
            protein_sequences=protein_sequences,
            batch_idx_map=batch_idx_map,
            structure_coords=structure_coords,
            go_aspects=go_aspects,
        )
        text_inputs_embeds = self._sanitize_embedding_tensor(text_inputs_embeds)
        return {
            "inputs_embeds": text_inputs_embeds,
            "attention_mask": attention_mask,
        }

    def forward_from_prepared_prefix(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        multimodal_cache: Optional[Dict[str, Any]] = None,
        prepared_inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            protein_sequences: List of protein sequence strings
            batch_idx_map: Batch mapping for protein sequences
            structure_coords: Optional tensor containing the structure coordinates
            labels: Labels for supervised fine-tuning
            go_aspects: GO aspects for protein sequences
            **kwargs: Additional arguments

        Returns:
            Outputs from the text model
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        if prepared_inputs_embeds is None:
            prepared = self.prepare_multimodal_prefix(
                input_ids=input_ids,
                attention_mask=attention_mask,
                protein_sequences=protein_sequences,
                batch_idx_map=batch_idx_map,
                structure_coords=structure_coords,
                go_aspects=go_aspects,
                multimodal_cache=multimodal_cache,
            )
            prepared_inputs_embeds = prepared["inputs_embeds"]

        outputs = self.forward_from_prepared_prefix(
            inputs_embeds=prepared_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        protein_sequences: Optional[List[str]] = None,
        batch_idx_map: Optional[List[int]] = None,
        structure_coords: Optional[torch.Tensor] = None,
        go_aspects: Optional[List[str]] = None,
        multimodal_cache: Optional[Dict[str, Any]] = None,
        prepared_inputs_embeds: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text based on protein and text inputs.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            protein_sequences: List of protein sequence strings
            batch_idx_map: Batch mapping for protein sequences
            structure_coords: Optional tensor containing the structure coordinates
            go_aspects: GO aspects for protein sequences
            **generation_kwargs: Additional arguments for generation

        Returns:
            Generated token IDs
        """
        prefer_original_generate = bool(generation_kwargs.pop("prefer_original_generate", False))
        do_sample = bool(generation_kwargs.get("do_sample", False))

        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        if prepared_inputs_embeds is None:
            prepared = self.prepare_multimodal_prefix(
                input_ids=input_ids,
                attention_mask=attention_mask,
                protein_sequences=protein_sequences,
                batch_idx_map=batch_idx_map,
                structure_coords=structure_coords,
                go_aspects=go_aspects,
                multimodal_cache=multimodal_cache,
            )
            prepared_inputs_embeds = prepared["inputs_embeds"]
        text_inputs_embeds = prepared_inputs_embeds

        # Generation with embeddings
        text_inputs_embeds = text_inputs_embeds.to(input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)

        # Unsloth's patched generate path is brittle with prompt embeddings. For training-time
        # sample traces we fall back to a small greedy decode loop that reuses the prepared
        # multimodal prefix embeddings and appends token embeddings directly.
        if prefer_original_generate and not do_sample:
            max_new_tokens = int(generation_kwargs.pop("max_new_tokens", 64))
            eos_token_id = generation_kwargs.pop("eos_token_id", getattr(self.text_model.config, "eos_token_id", None))
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0] if eos_token_id else None

            generated_ids = input_ids.clone()
            generated_attention_mask = attention_mask.clone()
            generated_inputs_embeds = text_inputs_embeds.clone()
            token_embedding_layer = self.text_model.get_input_embeddings()

            for _ in range(max_new_tokens):
                outputs = self.text_model(
                    inputs_embeds=generated_inputs_embeds,
                    attention_mask=generated_attention_mask,
                    use_cache=False,
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                next_token_embeds = token_embedding_layer(next_token).to(
                    dtype=generated_inputs_embeds.dtype,
                    device=generated_inputs_embeds.device,
                )
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                generated_inputs_embeds = torch.cat([generated_inputs_embeds, next_token_embeds], dim=1)
                next_mask = torch.ones(
                    (generated_attention_mask.size(0), 1),
                    dtype=generated_attention_mask.dtype,
                    device=generated_attention_mask.device,
                )
                generated_attention_mask = torch.cat([generated_attention_mask, next_mask], dim=1)

                if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                    break

            return generated_ids

        generate_fn = self.text_model.generate
        if prefer_original_generate:
            if hasattr(self.text_model, "_old_generate"):
                generate_fn = self.text_model._old_generate
            else:
                base_model = getattr(self.text_model, "base_model", None)
                if base_model is not None and hasattr(base_model, "_old_generate"):
                    generate_fn = base_model._old_generate

        generation_config = generation_kwargs.get("generation_config")
        if generation_config is None:
            base_generation_config = getattr(self.text_model, "generation_config", None)
            if base_generation_config is not None:
                generation_config = copy.deepcopy(base_generation_config)

        if generation_config is not None:
            # Avoid re-applying model-config defaults inside transformers.generate();
            # we explicitly control the per-call generation settings below.
            if hasattr(generation_config, "_from_model_config"):
                generation_config._from_model_config = False
            for field_name in (
                "do_sample",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "min_new_tokens",
                "max_new_tokens",
                "repetition_penalty",
                "pad_token_id",
                "eos_token_id",
            ):
                if field_name in generation_kwargs:
                    setattr(generation_config, field_name, generation_kwargs[field_name])
            if not bool(getattr(generation_config, "do_sample", do_sample)):
                base_generation_config = getattr(self.text_model, "generation_config", None)
                for field_name in ("temperature", "top_p", "top_k"):
                    if base_generation_config is not None and hasattr(base_generation_config, field_name):
                        setattr(generation_config, field_name, getattr(base_generation_config, field_name))
                if hasattr(generation_config, "min_p"):
                    generation_config.min_p = None
            elif getattr(generation_config, "temperature", None) is None:
                generation_config.temperature = 1.0
            generation_kwargs["generation_config"] = generation_config

        with torch.inference_mode():
            outputs = generate_fn(
                inputs_embeds=text_inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                **generation_kwargs,
            )

        return outputs  
