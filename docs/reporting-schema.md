# Reporting Schema

Explanations are serialized as JSON using the following structure:

```json
{
  "local": [
    {
      "sequence": "MKT...",
      "shap_values": [[0.1, -0.2, ...]],
      "interaction_values": [[[0.0, 0.1], ...]]
    }
  ],
  "global_": {
    "mean_abs_shap": {"0": 0.5, "1": 0.2}
  }
}
```

See `mapexploc.report` for the Pydantic models.
