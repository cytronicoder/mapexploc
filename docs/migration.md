# Migration from v0 to v1

* Package renamed from `explocal` to `mapexploc` with compatibility shims.
* New adapter interface decouples user models from the pipeline.
* SHAP explanations moved to `mapexploc.explainers` with automatic fallback.
* Added FastAPI server and React UI scaffolding.
* Reporting schema formalized with Pydantic models.
