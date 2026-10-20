"""PI0.5 training package. Cache utilities do not import the training stack."""
__version__ = "0.1.0"
def register_training_components():
    import importlib
    import transformers
    if transformers.__version__ != "5.3.0":
        raise RuntimeError("This training implementation requires transformers==5.3.0")
    for module in ("collators", "datasets.dataset_wrapper", "datasets.parquet_dataset_v3", "models", "optimizers", "tokenizers", "transforms"):
        importlib.import_module("fluxvla." + module)
