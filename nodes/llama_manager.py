import gc
import threading

from llama_cpp import Llama


class ManagedLlama:
    def __init__(self, config, llm):
        self.config = config
        self.llm = llm
        self.lock = threading.RLock()


_CACHE = {}
_OBJECT_CACHE = {}
_PATH_CACHE = {}
_CACHE_LOCK = threading.Lock()


def llama_config(ckpt_path, max_ctx, gpu_layers, n_threads, seed, use_mlock):
    return (
        ckpt_path,
        int(max_ctx),
        int(gpu_layers),
        int(n_threads),
        bool(use_mlock),
    )


def get_managed_llama(config, auto_load=True):
    stale = None
    with _CACHE_LOCK:
        cached = _CACHE.get(config)
        if cached is not None:
            return cached
        if not auto_load:
            return None

        ckpt_path, max_ctx, gpu_layers, n_threads, use_mlock = config
        stale = _PATH_CACHE.get(ckpt_path)
        if stale is not None:
            _CACHE.pop(stale.config, None)
            _OBJECT_CACHE.pop(id(stale.llm), None)
            _PATH_CACHE.pop(ckpt_path, None)

    if stale is not None:
        with stale.lock:
            close_llama(stale.llm)
        release_memory()

    llm = Llama(
        model_path=ckpt_path,
        offload_kqv=True,
        f16_kv=True,
        use_mlock=use_mlock,
        embedding=False,
        n_batch=1024,
        last_n_tokens_size=1024,
        verbose=True,
        seed=42,
        n_ctx=max_ctx,
        n_gpu_layers=gpu_layers,
        n_threads=n_threads,
        logits_all=True,
        echo=False,
    )
    cached = ManagedLlama(config, llm)
    with _CACHE_LOCK:
        _CACHE[config] = cached
        _OBJECT_CACHE[id(llm)] = cached
        _PATH_CACHE[ckpt_path] = cached
    return cached


def get_loaded_llama_by_path(ckpt_path):
    with _CACHE_LOCK:
        cached = _PATH_CACHE.get(ckpt_path)
        if cached is not None:
            return cached
    return None


def config_status(config):
    ckpt_path, max_ctx, gpu_layers, n_threads, use_mlock = config
    return {
        "path": ckpt_path,
        "max_ctx": max_ctx,
        "gpu_layers": gpu_layers,
        "n_threads": n_threads,
        "use_mlock": use_mlock,
    }


def managed_status(cached):
    return config_status(cached.config)


def loaded_llama_statuses():
    with _CACHE_LOCK:
        return [managed_status(cached) for cached in _CACHE.values()]


def managed_for_llama(llm):
    with _CACHE_LOCK:
        return _OBJECT_CACHE.get(id(llm))


def close_llama(llm):
    if llm is None:
        return
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def release_memory():
    gc.collect()
    try:
        import comfy.model_management
        comfy.model_management.soft_empty_cache()
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def unload_config(config):
    with _CACHE_LOCK:
        cached = _CACHE.pop(config, None)
        if cached is not None:
            _OBJECT_CACHE.pop(id(cached.llm), None)
            _PATH_CACHE.pop(cached.config[0], None)
    if cached is not None:
        with cached.lock:
            close_llama(cached.llm)
        release_memory()


def unload_all():
    with _CACHE_LOCK:
        cached_entries = list(_CACHE.values())
        _CACHE.clear()
        _OBJECT_CACHE.clear()
        _PATH_CACHE.clear()
    for cached in cached_entries:
        with cached.lock:
            close_llama(cached.llm)
    release_memory()
