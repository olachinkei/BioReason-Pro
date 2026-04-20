import fcntl
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class RegistryError(RuntimeError):
    """Raised when a research registry entry cannot be resolved."""


def normalize_text(value: Any) -> str:
    """Return a stable string representation for registry values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item not in (None, ""))
    return str(value)


def resolve_wandb_registry_path(payload: Mapping[str, Any]) -> str:
    """Return the configured W&B registry reference for an asset or source."""
    return normalize_text(
        payload.get("wandb_registry_path")
        or payload.get("artifact_path")
    ).strip()


def resolve_wandb_registry_url(payload: Mapping[str, Any]) -> str:
    """Return an optional human-facing W&B registry URL for documentation."""
    return normalize_text(payload.get("wandb_registry_url")).strip()


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_exported_env_file(path_value: str, *, override: bool = False) -> Dict[str, str]:
    """Load KEY=VALUE or export KEY=VALUE lines from a simple env file into os.environ."""
    loaded: Dict[str, str] = {}
    env_path = Path(path_value)
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            parsed_value = shlex.split(value, posix=True)
            normalized_value = parsed_value[0] if parsed_value else ""
        except ValueError:
            normalized_value = value.strip().strip('"').strip("'")

        if override or key not in os.environ:
            os.environ[key] = normalized_value
            loaded[key] = normalized_value

    return loaded


def expand_placeholders(value: Any) -> Any:
    """Expand environment variables recursively in registry data."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_placeholders(item) for key, item in value.items()}
    return value


def apply_template_context(value: Any, context: Mapping[str, Any]) -> Any:
    """Format registry strings with bundle-specific template values."""
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError:
            return value
    if isinstance(value, list):
        return [apply_template_context(item, context) for item in value]
    if isinstance(value, dict):
        return {key: apply_template_context(item, context) for key, item in value.items()}
    return value


def resolve_repo_path(path_value: str, repo_root: Path) -> str:
    """Resolve a possibly relative registry path against the repository root."""
    expanded = Path(expand_placeholders(path_value))
    if expanded.is_absolute():
        return str(expanded)
    runtime_root = normalize_text(os.getenv("BIOREASON_RUNTIME_ROOT")).strip()
    if runtime_root:
        parts = expanded.parts
        if len(parts) >= 2 and parts[0] == "data" and parts[1] == "artifacts":
            return str((Path(runtime_root).expanduser() / expanded).resolve())
    return str((repo_root / expanded).resolve())


def directory_has_content(path_value: str, required_paths: Optional[Sequence[str]] = None) -> bool:
    """Return True when a directory exists and has the required content."""
    directory = Path(path_value)
    if not directory.is_dir():
        return False

    if required_paths:
        return all((directory / rel_path).exists() for rel_path in required_paths)

    return any(directory.iterdir())


def ensure_directory(path_value: str) -> str:
    """Create a directory when it does not already exist."""
    Path(path_value).mkdir(parents=True, exist_ok=True)
    return path_value


def _artifact_lock_path(local_path: Path) -> Path:
    """Return the lock file used to serialize shared artifact downloads."""
    return local_path.parent / f".{local_path.name}.artifact.lock"


def _remove_download_target(path_value: Path) -> None:
    """Remove a download target regardless of whether it is a dir, file, or symlink."""
    if not path_value.exists() and not path_value.is_symlink():
        return
    if path_value.is_symlink() or path_value.is_file():
        path_value.unlink()
        return
    shutil.rmtree(path_value)


def _load_registry(path_value: str, repo_root: Path) -> Dict[str, Any]:
    registry_path = resolve_repo_path(path_value, repo_root)
    registry = expand_placeholders(load_json(registry_path))
    registry["_registry_path"] = registry_path
    return registry


def load_data_bundle(path_value: str, bundle_name: Optional[str], repo_root: Path) -> Dict[str, Any]:
    """Load and normalize a named data bundle from the research registry."""
    registry = _load_registry(path_value, repo_root)
    resolved_name = bundle_name or registry.get("default_bundle")
    bundles = registry.get("bundles", {})
    if resolved_name not in bundles:
        raise RegistryError(f"Unknown data bundle: {resolved_name}")

    bundle = dict(bundles[resolved_name])
    bundle["bundle_name"] = resolved_name
    bundle["registry_path"] = registry["_registry_path"]

    for asset_name in ("temporal_split_artifact", "reasoning_dataset", "ia_file"):
        asset = dict(bundle.get(asset_name, {}))
        if not asset:
            continue
        local_dir = asset.get("local_dir")
        if local_dir:
            asset["local_dir"] = resolve_repo_path(local_dir, repo_root)
        asset["wandb_registry_path"] = resolve_wandb_registry_path(asset)
        asset["wandb_registry_url"] = resolve_wandb_registry_url(asset)
        bundle[asset_name] = asset

    return bundle


def load_eval_target_registry(path_value: str, repo_root: Path) -> Dict[str, Any]:
    """Load the evaluation target registry."""
    return _load_registry(path_value, repo_root)


def expand_target_group(
    registry: Mapping[str, Any],
    target_name: Optional[str],
    target_group: Optional[str],
) -> List[str]:
    """Return the ordered list of targets to evaluate."""
    if bool(target_name) == bool(target_group):
        raise RegistryError("Specify exactly one of --target or --target-group.")

    if target_name:
        return [target_name]

    groups = registry.get("target_groups", {})
    if target_group not in groups:
        raise RegistryError(f"Unknown target group: {target_group}")
    return list(groups[target_group])


def load_eval_target(
    registry: Mapping[str, Any],
    target_name: str,
    repo_root: Path,
) -> Dict[str, Any]:
    """Load and normalize a single evaluation target definition."""
    targets = registry.get("targets", {})
    if target_name not in targets:
        raise RegistryError(f"Unknown evaluation target: {target_name}")

    target = dict(targets[target_name])
    target["target_name"] = target_name
    target["registry_path"] = registry.get("_registry_path")

    for source_list_key in ("model_sources", "prediction_sources"):
        sources = []
        for source in target.get(source_list_key, []):
            source_copy = dict(source)
            local_dir = source_copy.get("local_dir")
            if local_dir:
                source_copy["local_dir"] = resolve_repo_path(local_dir, repo_root)
            local_path = source_copy.get("local_path")
            if local_path:
                source_copy["local_path"] = resolve_repo_path(local_path, repo_root)
            source_copy["wandb_registry_path"] = resolve_wandb_registry_path(source_copy)
            source_copy["wandb_registry_url"] = resolve_wandb_registry_url(source_copy)
            sources.append(source_copy)
        if sources:
            target[source_list_key] = sources

    return target


def get_wandb_artifact_metadata(wandb_registry_path: str) -> Dict[str, Any]:
    """Fetch W&B artifact metadata without assuming it is already downloaded."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RegistryError("wandb is required to inspect registry artifacts.") from exc

    api = wandb.Api()
    artifact = api.artifact(wandb_registry_path)
    return dict(getattr(artifact, "metadata", {}) or {})


def download_wandb_artifact(
    wandb_registry_path: str,
    local_dir: str,
    required_paths: Optional[Sequence[str]] = None,
) -> str:
    """Download a W&B artifact into a deterministic local directory."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RegistryError("wandb is required to download registry artifacts.") from exc

    local_path = Path(local_dir)
    ensure_directory(str(local_path.parent))
    lock_path = _artifact_lock_path(local_path)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if directory_has_content(local_dir, required_paths):
            return local_dir
        _remove_download_target(local_path)
        ensure_directory(local_dir)
        api = wandb.Api()
        artifact = api.artifact(wandb_registry_path)
        artifact.download(root=local_dir)
        if required_paths and not directory_has_content(local_dir, required_paths):
            raise RegistryError(
                "Downloaded W&B artifact is missing required paths. "
                f"artifact={wandb_registry_path} required_paths={list(required_paths)}"
            )
    return local_dir


def download_huggingface_snapshot(
    repo_id: str,
    local_dir: str,
    revision: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
) -> str:
    """Download a Hugging Face snapshot into a deterministic local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RegistryError("huggingface_hub is required to download model snapshots.") from exc

    ensure_directory(local_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
    )
    return local_dir


def _resolve_repo_id(source: Mapping[str, Any]) -> str:
    repo_id = normalize_text(source.get("repo_id")).strip()
    if repo_id:
        return repo_id

    repo_id_env = normalize_text(source.get("repo_id_env")).strip()
    if repo_id_env:
        return os.getenv(repo_id_env, "").strip()
    return ""


def materialize_source(
    source: Mapping[str, Any],
    *,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    """Resolve a local path for a single model or prediction source."""
    source_type = normalize_text(source.get("type")).strip()
    required_paths = list(source.get("required_paths", []))

    if source_type == "local_path":
        local_path = normalize_text(source.get("local_path")).strip()
        if not local_path:
            if required:
                raise RegistryError("local_path source is missing local_path.")
            return None
        if not Path(local_path).exists():
            if required:
                raise RegistryError(f"Required local path does not exist: {local_path}")
            return None
        return {
            "source_type": source_type,
            "local_path": local_path,
            "source_ref": local_path,
            "downloaded": False,
        }

    local_dir = normalize_text(source.get("local_dir")).strip()
    if not local_dir:
        if required:
            raise RegistryError(f"{source_type} source is missing local_dir.")
        return None

    if directory_has_content(local_dir, required_paths):
        resolved_repo_id = _resolve_repo_id(source)
        source_ref = normalize_text(
            resolve_wandb_registry_path(source)
            or (f"hf://{resolved_repo_id}" if source_type == "huggingface" and resolved_repo_id else "")
            or source.get("repo_id")
            or source.get("local_path")
            or source.get("local_dir")
        )
        return {
            "source_type": source_type,
            "local_path": local_dir,
            "source_ref": source_ref,
            "downloaded": False,
            "wandb_registry_path": resolve_wandb_registry_path(source),
            "wandb_registry_url": resolve_wandb_registry_url(source),
        }

    if source_type in {"local_dir", "directory"}:
        if not local_dir:
            if required:
                raise RegistryError(f"{source_type} source is missing local_dir.")
            return None
        if directory_has_content(local_dir, required_paths):
            return {
                "source_type": source_type,
                "local_path": local_dir,
                "source_ref": local_dir,
                "downloaded": False,
                "wandb_registry_path": resolve_wandb_registry_path(source),
                "wandb_registry_url": resolve_wandb_registry_url(source),
            }
        if required:
            raise RegistryError(f"Required local directory does not exist or is incomplete: {local_dir}")
        return None

    if source_type == "wandb_artifact":
        wandb_registry_path = resolve_wandb_registry_path(source)
        if not wandb_registry_path:
            if required:
                raise RegistryError("wandb_artifact source is missing wandb_registry_path.")
            return None
        download_wandb_artifact(wandb_registry_path, local_dir, required_paths=required_paths)
        return {
            "source_type": source_type,
            "local_path": local_dir,
            "source_ref": wandb_registry_path,
            "downloaded": True,
            "wandb_registry_path": wandb_registry_path,
            "wandb_registry_url": resolve_wandb_registry_url(source),
        }

    if source_type == "huggingface":
        repo_id = _resolve_repo_id(source)
        if not repo_id:
            if required:
                repo_id_env = normalize_text(source.get("repo_id_env")).strip()
                if repo_id_env:
                    raise RegistryError(
                        f"Hugging Face repo_id is not configured. Set environment variable {repo_id_env}."
                    )
                raise RegistryError("huggingface source is missing repo_id.")
            return None
        download_huggingface_snapshot(
            repo_id=repo_id,
            local_dir=local_dir,
            revision=normalize_text(source.get("revision")).strip() or None,
            allow_patterns=source.get("allow_patterns"),
            ignore_patterns=source.get("ignore_patterns"),
        )
        return {
            "source_type": source_type,
            "local_path": local_dir,
            "source_ref": f"hf://{repo_id}",
            "downloaded": True,
            "wandb_registry_path": resolve_wandb_registry_path(source),
            "wandb_registry_url": resolve_wandb_registry_url(source),
        }

    if required:
        raise RegistryError(f"Unsupported source type: {source_type}")
    return None


def materialize_first_available_source(
    sources: Sequence[Mapping[str, Any]],
    *,
    allow_missing: bool = False,
) -> Optional[Dict[str, Any]]:
    """Resolve the first usable source from a registry source list."""
    if not sources:
        if allow_missing:
            return None
        raise RegistryError("No sources are configured.")

    errors: List[str] = []
    for source in sources:
        try:
            resolved = materialize_source(source, required=not allow_missing)
        except RegistryError as exc:
            errors.append(str(exc))
            continue
        if resolved:
            return resolved

    if allow_missing:
        return None

    raise RegistryError("; ".join(errors) or "No configured source could be resolved.")


def materialize_bundle_asset(asset: Mapping[str, Any]) -> Dict[str, Any]:
    """Download a W&B-backed data asset when needed and return local metadata."""
    local_dir = normalize_text(asset.get("local_dir")).strip()
    wandb_registry_path = resolve_wandb_registry_path(asset)
    required_paths = list(asset.get("required_paths", []))
    artifact_metadata: Dict[str, Any] = {}
    dataset_source = normalize_text(asset.get("dataset_source"))
    dataset_name = normalize_text(asset.get("dataset_name"))

    if local_dir and directory_has_content(local_dir, required_paths):
        return {
            "local_dir": local_dir,
            "wandb_registry_path": wandb_registry_path,
            "wandb_registry_url": resolve_wandb_registry_url(asset),
            "downloaded": False,
            "dataset_source": dataset_source,
            "dataset_name": dataset_name,
            "artifact_metadata": artifact_metadata,
        }

    if wandb_registry_path and (not dataset_source or not dataset_name):
        try:
            artifact_metadata = get_wandb_artifact_metadata(wandb_registry_path)
        except RegistryError:
            artifact_metadata = {}
        dataset_source = normalize_text(dataset_source or artifact_metadata.get("dataset_source"))
        dataset_name = normalize_text(dataset_name or artifact_metadata.get("dataset_name"))

    if wandb_registry_path and local_dir:
        download_wandb_artifact(wandb_registry_path, local_dir, required_paths=required_paths)
        return {
            "local_dir": local_dir,
            "wandb_registry_path": wandb_registry_path,
            "wandb_registry_url": resolve_wandb_registry_url(asset),
            "downloaded": True,
            "dataset_source": dataset_source,
            "dataset_name": dataset_name,
            "artifact_metadata": artifact_metadata,
        }

    return {
        "local_dir": local_dir,
        "wandb_registry_path": wandb_registry_path,
        "wandb_registry_url": resolve_wandb_registry_url(asset),
        "downloaded": False,
        "dataset_source": dataset_source,
        "dataset_name": dataset_name,
        "artifact_metadata": artifact_metadata,
    }
