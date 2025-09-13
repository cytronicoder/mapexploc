# Add Your Own Model

MAP-ExPLoc wraps any sequence-to-localization model via a lightweight adapter
interface. Implement `predict` and `predict_proba` on batches of sequences and
(optionally) `embed`.

```python
from mapexploc.adapter import BaseModelAdapter

class MyModelAdapter(BaseModelAdapter):
    def __init__(self, model):
        self.model = model

    def predict(self, batch):
        return self.model.predict(batch)

    def predict_proba(self, batch):
        return self.model.predict_proba(batch)
```

Register the adapter using `load_adapter` or pass it directly to the API.
