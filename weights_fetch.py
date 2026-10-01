"""Fetch model weights from Hugging Face on first use.

The weights are not committed to this repository -- they are ~335 MB and would
dominate every clone and, on GitHub LFS, burn the owner's bandwidth quota on the
first couple of dozen installs. They live in a Hugging Face dataset instead:

    https://huggingface.co/zjkhurry/comfyui-docres-weights

Downloads are cached next to the node in `weights/`, so each file is fetched
once. A file already present is never re-downloaded, and a download that was
interrupted is retried on the next run rather than failing at load time.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

HF_REPO = "zjkhurry/comfyui-docres-weights"
WEIGHT_DIR = os.path.join(HERE, "weights")

# filename -> filename on the Hub
FILES = {
    "docres.safetensors": "docres.safetensors",
    "mbd.safetensors": "mbd.safetensors",
    "ddc_fiducial1024_v1.safetensors": "ddc_fiducial1024_v1.safetensors",
}


def _local_path(name):
    """Where `name` lives once it has been downloaded."""
    return os.path.join(WEIGHT_DIR, name)


def _download(name, repo_id, local):
    """Download one file straight into `local`.

    `local_dir` puts the file where we want it instead of a shared cache that
    would then have to be copied, which would leave a second copy on disk for no
    benefit. A partial download is discarded so a retry starts clean.
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(WEIGHT_DIR, exist_ok=True)
    partial = local + ".part"
    if os.path.exists(partial):
        os.remove(partial)
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=name, local_dir=WEIGHT_DIR,
        )
    except Exception:
        if os.path.exists(partial):
            os.remove(partial)
        raise
    # hf_hub_download returns the real path; normalise it to what we promised.
    return local if os.path.isfile(local) else path


def ensure(name, repo_id=HF_REPO):
    """Return a local path for `name`, downloading it first if necessary.

    An existing file is used as-is, so a user who dropped the weights in
    manually, or who is offline after the first run, is not forced to fetch
    anything again.
    """
    local = _local_path(name)
    if os.path.isfile(local) and os.path.getsize(local) > 0:
        return local
    return _download(name, repo_id, local)


def prefetch(names=None, repo_id=HF_REPO):
    """Download every weight in `FILES` (or just `names`).

    Safe to call when the files are already present; it is how an installer or a
    first-run hook warms the cache without loading any model.
    """
    wanted = list(names) if names else list(FILES)
    missing = [n for n in wanted if not (
        os.path.isfile(_local_path(n)) and os.path.getsize(_local_path(n)) > 0
    )]
    for name in missing:
        _download(name, repo_id, _local_path(name))
    return [_local_path(n) for n in wanted]
