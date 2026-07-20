"""Deferred Qwen3.5 compatibility setup for VERL Ray training workers."""


def install_qwen35_worker_patches():
    from sitecustomize import install_attention_patch

    install_attention_patch()
