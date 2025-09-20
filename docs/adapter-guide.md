# Custom Model Adapter Guide

MAP-ExPLoc provides a flexible adapter interface that allows you to integrate any sequence-to-localization model into the MAP-ExPLoc ecosystem. This guide covers how to create custom adapters, implement required methods, and integrate your models seamlessly.

## Overview

The adapter interface standardizes how different models interact with MAP-ExPLoc, enabling:

- **Unified API**: Consistent interface across different model types
- **Batch Processing**: Efficient handling of multiple sequences
- **Interpretability**: Integration with explainability tools
- **Extensibility**: Easy addition of new model architectures

## Basic Adapter Interface

### Required Methods

All adapters must inherit from `BaseModelAdapter` and implement these core methods:

```python
from mapexploc.adapter import BaseModelAdapter
import numpy as np

class MyModelAdapter(BaseModelAdapter):
    def __init__(self, model, class_names=None):
        """Initialize the adapter with your model.

        Args:
            model: Your trained model instance
            class_names: List of class names (optional)
        """
        self.model = model
        self.class_names = class_names or self._get_default_classes()

    def predict(self, sequences):
        """Predict class labels for a batch of sequences.

        Args:
            sequences: List of protein sequences (strings)

        Returns:
            numpy.ndarray: Array of predicted class labels
        """
        return self.model.predict(sequences)

    def predict_proba(self, sequences):
        """Predict class probabilities for a batch of sequences.

        Args:
            sequences: List of protein sequences (strings)

        Returns:
            numpy.ndarray: Array of shape (n_samples, n_classes) with probabilities
        """
        return self.model.predict_proba(sequences)

    def _get_default_classes(self):
        """Return default class names if not provided."""
        return [
            'Cell Surface', 'Cytoplasm', 'Endoplasmic Reticulum',
            'Golgi Apparatus', 'Lysosome', 'Membrane', 'Mitochondrion',
            'Nucleus', 'Other', 'Periplasm', 'Peroxisome', 'Plastid',
            'Secreted', 'Vacuole', 'Virion'
        ]
```

### Optional Methods

For enhanced functionality, you can implement these optional methods:

```python
class AdvancedModelAdapter(BaseModelAdapter):
    def embed(self, sequences):
        """Generate embeddings for sequences (optional).

        Args:
            sequences: List of protein sequences

        Returns:
            numpy.ndarray: Embedding vectors of shape (n_samples, embedding_dim)
        """
        if hasattr(self.model, 'embed'):
            return self.model.embed(sequences)
        else:
            # Fallback to feature extraction if embeddings not available
            return self._extract_features(sequences)

    def get_feature_names(self):
        """Return feature names for interpretability."""
        if hasattr(self.model, 'feature_names_'):
            return self.model.feature_names_
        return [f'feature_{i}' for i in range(self.get_n_features())]

    def get_n_features(self):
        """Return number of features."""
        if hasattr(self.model, 'n_features_'):
            return self.model.n_features_
        return 100  # Default fallback

    def supports_batch_processing(self):
        """Check if model supports batch processing efficiently."""
        return hasattr(self.model, 'predict_batch')

    def get_model_info(self):
        """Return model metadata."""
        return {
            'model_type': type(self.model).__name__,
            'version': getattr(self.model, 'version', '1.0'),
            'classes': self.class_names,
            'features': self.get_n_features()
        }
```

## Specific Model Integrations

### Scikit-learn Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from mapexploc.preprocessing import ProteinFeatureExtractor

class SklearnAdapter(BaseModelAdapter):
    def __init__(self, sklearn_model, feature_extractor=None):
        self.model = sklearn_model
        self.feature_extractor = feature_extractor or ProteinFeatureExtractor()

        # Get class names from model if available
        if hasattr(sklearn_model, 'classes_'):
            self.class_names = sklearn_model.classes_.tolist()
        else:
            self.class_names = self._get_default_classes()

    def _prepare_features(self, sequences):
        """Convert sequences to feature vectors."""
        return self.feature_extractor.transform(sequences)

    def predict(self, sequences):
        features = self._prepare_features(sequences)
        return self.model.predict(features)

    def predict_proba(self, sequences):
        features = self._prepare_features(sequences)
        return self.model.predict_proba(features)

    def embed(self, sequences):
        """Use feature vectors as embeddings."""
        return self._prepare_features(sequences)

# Example usage
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# ... train your model ...
adapter = SklearnAdapter(rf_model)
```

### PyTorch Models

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class PyTorchAdapter(BaseModelAdapter):
    def __init__(self, pytorch_model, tokenizer=None, device='cpu'):
        self.model = pytorch_model
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('Rostlab/prot_bert')
        self.device = device
        self.model.to(device)

        # Assume model has a class_names attribute or use default
        self.class_names = getattr(pytorch_model, 'class_names', self._get_default_classes())

    def _tokenize_sequences(self, sequences):
        """Tokenize protein sequences."""
        # Add spaces between amino acids for BERT-style models
        spaced_sequences = [' '.join(list(seq)) for seq in sequences]

        return self.tokenizer(
            spaced_sequences,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(self.device)

    def predict(self, sequences):
        self.model.eval()
        with torch.no_grad():
            inputs = self._tokenize_sequences(sequences)
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            return predictions.cpu().numpy()

    def predict_proba(self, sequences):
        self.model.eval()
        with torch.no_grad():
            inputs = self._tokenize_sequences(sequences)
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            return probabilities.cpu().numpy()

    def embed(self, sequences):
        """Extract embeddings from model."""
        self.model.eval()
        with torch.no_grad():
            inputs = self._tokenize_sequences(sequences)

            # Get hidden states (assuming BERT-like model)
            if hasattr(self.model, 'bert') or hasattr(self.model, 'encoder'):
                encoder = getattr(self.model, 'bert', getattr(self.model, 'encoder', None))
                outputs = encoder(**inputs)
                # Use [CLS] token or mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            else:
                # Extract from penultimate layer
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-2].mean(dim=1)

            return embeddings.cpu().numpy()

# Example usage
class ProteinBERTClassifier(nn.Module):
    def __init__(self, n_classes=15):
        super().__init__()
        self.bert = AutoModel.from_pretrained('Rostlab/prot_bert')
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return type('Outputs', (), {'logits': logits})()

# Initialize and use adapter
model = ProteinBERTClassifier()
# ... train your model ...
adapter = PyTorchAdapter(model, device='cuda' if torch.cuda.is_available() else 'cpu')
```

### TensorFlow/Keras Models

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TensorFlowAdapter(BaseModelAdapter):
    def __init__(self, tf_model, tokenizer=None, max_length=1000):
        self.model = tf_model
        self.max_length = max_length

        # Create or load tokenizer
        if tokenizer is None:
            self.tokenizer = self._create_tokenizer()
        else:
            self.tokenizer = tokenizer

        # Get class names
        self.class_names = getattr(tf_model, 'class_names', self._get_default_classes())

    def _create_tokenizer(self):
        """Create amino acid tokenizer."""
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        tokenizer = Tokenizer(char_level=True, oov_token='<UNK>')
        tokenizer.fit_on_texts(amino_acids)
        return tokenizer

    def _prepare_sequences(self, sequences):
        """Convert sequences to tokenized arrays."""
        # Tokenize sequences
        tokenized = self.tokenizer.texts_to_sequences(sequences)

        # Pad sequences
        padded = pad_sequences(tokenized, maxlen=self.max_length, padding='post')

        return padded

    def predict(self, sequences):
        prepared = self._prepare_sequences(sequences)
        predictions = self.model.predict(prepared, verbose=0)
        return predictions.argmax(axis=-1)

    def predict_proba(self, sequences):
        prepared = self._prepare_sequences(sequences)
        return self.model.predict(prepared, verbose=0)

    def embed(self, sequences):
        """Extract embeddings from intermediate layer."""
        prepared = self._prepare_sequences(sequences)

        # Create model that outputs embeddings (assumes embedding layer exists)
        if hasattr(self.model, 'layers'):
            # Find embedding layer or use penultimate layer
            embedding_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.layers[-2].output  # Penultimate layer
            )
            return embedding_model.predict(prepared, verbose=0)

        # Fallback to prediction probabilities as features
        return self.predict_proba(sequences)

# Example usage with CNN model
def create_cnn_model(vocab_size, max_length, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    return model

cnn_model = create_cnn_model(vocab_size=25, max_length=1000, n_classes=15)
# ... train your model ...
adapter = TensorFlowAdapter(cnn_model)
```

## Advanced Adapter Features

### Caching and Performance Optimization

```python
from functools import lru_cache
import pickle
import hashlib

class CachedAdapter(BaseModelAdapter):
    def __init__(self, base_adapter, cache_size=1000):
        self.base_adapter = base_adapter
        self.cache_size = cache_size
        self._prediction_cache = {}
        self._embedding_cache = {}

    def _hash_sequences(self, sequences):
        """Create hash for sequence batch."""
        sequence_str = '|'.join(sorted(sequences))
        return hashlib.md5(sequence_str.encode()).hexdigest()

    @lru_cache(maxsize=1000)
    def _cached_predict(self, sequence_hash, sequences_tuple):
        """Cached prediction for single sequences."""
        sequences = list(sequences_tuple)
        return self.base_adapter.predict(sequences)

    def predict(self, sequences):
        # For small batches, use individual caching
        if len(sequences) <= 10:
            results = []
            for seq in sequences:
                seq_hash = hashlib.md5(seq.encode()).hexdigest()
                result = self._cached_predict(seq_hash, (seq,))
                results.append(result[0])
            return np.array(results)

        # For large batches, use batch prediction
        batch_hash = self._hash_sequences(sequences)
        if batch_hash not in self._prediction_cache:
            self._prediction_cache[batch_hash] = self.base_adapter.predict(sequences)

        return self._prediction_cache[batch_hash]

    def predict_proba(self, sequences):
        # Similar caching strategy for probabilities
        batch_hash = self._hash_sequences(sequences)
        cache_key = f"proba_{batch_hash}"

        if cache_key not in self._prediction_cache:
            self._prediction_cache[cache_key] = self.base_adapter.predict_proba(sequences)

        return self._prediction_cache[cache_key]

    def clear_cache(self):
        """Clear all caches."""
        self._prediction_cache.clear()
        self._embedding_cache.clear()
        self._cached_predict.cache_clear()
```

### Multi-Model Ensemble Adapter

```python
class EnsembleAdapter(BaseModelAdapter):
    def __init__(self, adapters, weights=None, voting='soft'):
        """Ensemble of multiple adapters.

        Args:
            adapters: List of BaseModelAdapter instances
            weights: Optional weights for each adapter
            voting: 'soft' for probability averaging, 'hard' for majority vote
        """
        self.adapters = adapters
        self.weights = weights or [1.0] * len(adapters)
        self.voting = voting

        # Use class names from first adapter
        self.class_names = adapters[0].class_names

    def predict(self, sequences):
        if self.voting == 'soft':
            # Use probability-based voting
            probas = self.predict_proba(sequences)
            return probas.argmax(axis=1)
        else:
            # Hard voting
            predictions = []
            for adapter, weight in zip(self.adapters, self.weights):
                pred = adapter.predict(sequences)
                predictions.append(pred)

            # Majority vote with weights
            predictions = np.array(predictions)
            weighted_votes = np.zeros((len(sequences), len(self.class_names)))

            for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
                for j, class_pred in enumerate(pred):
                    weighted_votes[j, class_pred] += weight

            return weighted_votes.argmax(axis=1)

    def predict_proba(self, sequences):
        probabilities = []

        for adapter, weight in zip(self.adapters, self.weights):
            proba = adapter.predict_proba(sequences)
            probabilities.append(proba * weight)

        # Average weighted probabilities
        ensemble_proba = np.mean(probabilities, axis=0)

        # Renormalize
        return ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)

    def embed(self, sequences):
        """Concatenate embeddings from all adapters."""
        embeddings = []

        for adapter in self.adapters:
            if hasattr(adapter, 'embed'):
                emb = adapter.embed(sequences)
                embeddings.append(emb)

        if embeddings:
            return np.concatenate(embeddings, axis=1)
        else:
            # Fallback to probability features
            return self.predict_proba(sequences)

# Example: Combine different model types
rf_adapter = SklearnAdapter(trained_rf_model)
cnn_adapter = TensorFlowAdapter(trained_cnn_model)
transformer_adapter = PyTorchAdapter(trained_transformer_model)

ensemble = EnsembleAdapter([rf_adapter, cnn_adapter, transformer_adapter],
                          weights=[0.3, 0.3, 0.4])
```

## Registration and Usage

### Registering Custom Adapters

```python
from mapexploc import register_adapter, load_adapter

# Register your adapter
register_adapter('my_custom_model', MyModelAdapter)

# Load and use
adapter = load_adapter('my_custom_model', model=your_trained_model)

# Or register with instance
custom_instance = MyModelAdapter(your_model)
register_adapter('my_instance', custom_instance)
```

### Integration with MAP-ExPLoc API

```python
from mapexploc import Predictor, Explainer

# Use custom adapter with predictor
predictor = Predictor(adapter=custom_adapter)

# Make predictions
sequences = ["MKTIIALSYIFCLVFADYKDDDDK", "MALWMRLLPLLALLALWGPGPGGA"]
predictions = predictor.predict(sequences)

# Use with explainer
explainer = Explainer(adapter=custom_adapter, method='shap')
explanations = explainer.explain(sequences)
```

### Configuration File Integration

Create a configuration file for your custom adapter:

```yaml
# config/custom_adapter.yml
adapter:
  type: my_custom_model
  parameters:
    model_path: models/my_trained_model.pkl
    feature_extractor:
      type: protein_features
      features:
        - amino_acid_composition
        - dipeptide_composition
        - molecular_weight

prediction:
  batch_size: 32
  confidence_threshold: 0.7

explanation:
  method: shap
  background_size: 100
```

Load with configuration:

```python
from mapexploc import Config, create_adapter_from_config

config = Config.from_file('config/custom_adapter.yml')
adapter = create_adapter_from_config(config.adapter)
```

## Testing Your Adapter

### Unit Testing Framework

```python
import unittest
import numpy as np
from mapexploc.testing import AdapterTestSuite

class TestMyAdapter(AdapterTestSuite):
    def setUp(self):
        # Initialize your adapter
        self.adapter = MyModelAdapter(your_model)
        self.test_sequences = [
            "MKTIIALSYIFCLVFADYKDDDDK",
            "MALWMRLLPLLALLALWGPGPGGA",
            "MVLSEGEWQLVLHVWAKVEADVAGHG"
        ]

    def test_predict_output_shape(self):
        """Test prediction output shape."""
        predictions = self.adapter.predict(self.test_sequences)
        self.assertEqual(len(predictions), len(self.test_sequences))

    def test_predict_proba_output_shape(self):
        """Test probability output shape."""
        probabilities = self.adapter.predict_proba(self.test_sequences)
        expected_shape = (len(self.test_sequences), len(self.adapter.class_names))
        self.assertEqual(probabilities.shape, expected_shape)

    def test_probability_sums(self):
        """Test that probabilities sum to 1."""
        probabilities = self.adapter.predict_proba(self.test_sequences)
        np.testing.assert_array_almost_equal(probabilities.sum(axis=1), 1.0)

    def test_consistent_predictions(self):
        """Test prediction consistency."""
        pred1 = self.adapter.predict(self.test_sequences)
        pred2 = self.adapter.predict(self.test_sequences)
        np.testing.assert_array_equal(pred1, pred2)

if __name__ == '__main__':
    unittest.main()
```

### Performance Benchmarking

```python
import time
from mapexploc.benchmarking import AdapterBenchmark

def benchmark_adapter(adapter, test_sequences):
    """Benchmark adapter performance."""

    benchmark = AdapterBenchmark(adapter)

    # Test different batch sizes
    batch_sizes = [1, 10, 50, 100]
    results = {}

    for batch_size in batch_sizes:
        # Create test batch
        test_batch = test_sequences[:batch_size] * (batch_size // len(test_sequences) + 1)
        test_batch = test_batch[:batch_size]

        # Benchmark prediction
        start_time = time.time()
        predictions = adapter.predict(test_batch)
        pred_time = time.time() - start_time

        # Benchmark probabilities
        start_time = time.time()
        probabilities = adapter.predict_proba(test_batch)
        proba_time = time.time() - start_time

        results[batch_size] = {
            'predict_time': pred_time,
            'predict_throughput': batch_size / pred_time,
            'proba_time': proba_time,
            'proba_throughput': batch_size / proba_time
        }

    return results

# Run benchmark
results = benchmark_adapter(your_adapter, test_sequences)
for batch_size, metrics in results.items():
    print(f"Batch size {batch_size}:")
    print(f"  Prediction: {metrics['predict_throughput']:.2f} sequences/second")
    print(f"  Probabilities: {metrics['proba_throughput']:.2f} sequences/second")
```

## Best Practices

### Error Handling

```python
class RobustAdapter(BaseModelAdapter):
    def predict(self, sequences):
        try:
            # Validate input
            self._validate_sequences(sequences)

            # Make predictions
            return self.model.predict(sequences)

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return default predictions or raise informative error
            raise ValueError(f"Prediction failed for {len(sequences)} sequences: {e}")

    def _validate_sequences(self, sequences):
        """Validate input sequences."""
        if not sequences:
            raise ValueError("Empty sequence list provided")

        for i, seq in enumerate(sequences):
            if not isinstance(seq, str):
                raise TypeError(f"Sequence {i} is not a string: {type(seq)}")

            if len(seq) == 0:
                raise ValueError(f"Empty sequence at index {i}")

            # Check for valid amino acids (optional)
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
            invalid_chars = set(seq.upper()) - valid_aa
            if invalid_chars:
                self.logger.warning(f"Invalid characters in sequence {i}: {invalid_chars}")
```

### Memory Management

```python
class MemoryEfficientAdapter(BaseModelAdapter):
    def __init__(self, model, max_batch_size=100):
        self.model = model
        self.max_batch_size = max_batch_size

    def predict(self, sequences):
        """Process sequences in chunks to manage memory."""
        if len(sequences) <= self.max_batch_size:
            return self.model.predict(sequences)

        # Process in chunks
        results = []
        for i in range(0, len(sequences), self.max_batch_size):
            chunk = sequences[i:i + self.max_batch_size]
            chunk_results = self.model.predict(chunk)
            results.extend(chunk_results)

        return np.array(results)
```

### Logging and Monitoring

```python
import logging

class LoggingAdapter(BaseModelAdapter):
    def __init__(self, base_adapter):
        self.base_adapter = base_adapter
        self.logger = logging.getLogger(f"{__name__}.{type(base_adapter).__name__}")

    def predict(self, sequences):
        self.logger.info(f"Predicting {len(sequences)} sequences")
        start_time = time.time()

        try:
            results = self.base_adapter.predict(sequences)
            duration = time.time() - start_time
            self.logger.info(f"Prediction completed in {duration:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
```

## Conclusion

The MAP-ExPLoc adapter interface provides a powerful and flexible way to integrate any protein localization model into the MAP-ExPLoc ecosystem. By implementing the required methods and following the best practices outlined in this guide, you can:

- Seamlessly integrate existing models
- Leverage MAP-ExPLoc's explainability tools
- Build ensemble methods combining multiple approaches
- Create production-ready, robust model deployments

For additional examples and advanced use cases, see the `examples/adapters/` directory in the MAP-ExPLoc repository.
