# Choosing an Explainer

`ShapExplainer` automatically selects `DeepExplainer` when the adapter exposes
embeddings; otherwise it falls back to `KernelExplainer`.

Provide a background dataset representative of your domain to obtain stable
attributions. For very long sequences, subsampling and caching are performed
internally.
