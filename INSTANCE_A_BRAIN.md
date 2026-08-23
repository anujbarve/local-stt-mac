# Instance A: The Brain (`brain/`)

## Your Mission

Build the intelligence layer — the custom neural network that understands text. You produce `Understanding` objects that the vault (Instance B) consumes.

**You are the brain. You think. You classify. You extract. You learn.**

## Your Directory

```
brain/
├── __init__.py
├── config.py          # Hyperparameters, type definitions
├── tokenizer.py       # Custom tokenizer for spoken text
├── backbone.py        # Shared transformer backbone
├── heads.py           # All 10 task heads
├── model.py           # Full multi-task model
├── embedder.py        # sentence-transformers wrapper
├── entity_linker.py   # Entity resolution across notes
├── data.py            # Synthetic data generator
├── train.py           # Training loop
├── understand.py      # Main inference API ← THIS IS YOUR ENTRY POINT
├── correct.py         # Correction logging
└── tests/
    ├── test_model.py
    ├── test_data.py
    ├── test_understand.py
    └── test_tokenizer.py
```

## Rules

1. **DO NOT** touch `transcribe.py`, `vault/`, or any existing voiceassist module
2. **DO NOT** modify `brain_contract.py` after initial setup
3. **DO** import types from `brain_contract.py` — never redefine them
4. **DO** keep your modules self-contained — no side effects on import
5. **DO** write tests for every module

## Shared Types

Import from `brain_contract.py` at the project root:

```python
from brain_contract import (
    Understanding, Entity, TemporalInfo,
    NoteType, EntityType, Priority, Urgency, Sentiment,
    QueryResult, Correction, TOPICS, RELATION_TYPES,
)
```

## Module Specifications

---

### 1. `brain/config.py` — Configuration

```python
"""All hyperparameters and constants for the brain."""

# Model architecture
EMBEDDING_DIM = 256          # backbone hidden dimension
NUM_LAYERS = 4               # transformer layers in backbone
NUM_HEADS = 4                # attention heads
FF_DIM = 1024                # feed-forward intermediate dimension
DROPOUT = 0.2
MAX_SEQ_LEN = 512            # maximum input token length

# Vocabulary
VOCAB_SIZE = 30_000          # tokenizer vocabulary size

# Head dimensions
CLASSIFIER_HIDDEN = 128
ENTITY_HIDDEN = 128
PRIORITY_HIDDEN = 64
URGENCY_HIDDEN = 64
SENTIMENT_HIDDEN = 64
ACTION_DECODER_LAYERS = 2
ACTION_DECODER_HEADS = 4
ACTION_MAX_LEN = 32          # max action text tokens
SUMMARY_HIDDEN = 64
TEMPORAL_HIDDEN = 64
TOPIC_HIDDEN = 128
RELATION_HIDDEN = 128

# Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 20
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
PRIORITY_LOSS_WEIGHT = 1.0
ENTITY_LOSS_WEIGHT = 2.0     # entities are harder, weight more
CLASSIFIER_LOSS_WEIGHT = 1.0
TOPIC_LOSS_WEIGHT = 0.5      # multi-label, lower weight

# Synthetic data
SYNTHETIC_SAMPLES = 15_000
SYNTHETIC_SEED = 42

# Checkpoints
CHECKPOINT_DIR = "checkpoints/brain"
CHECKPOINT_EVERY = 5         # save every N epochs
```

---

### 2. `brain/tokenizer.py` — Custom Tokenizer

**Purpose:** Tokenize spoken transcriptions into integer sequences.

**Requirements:**
- Build vocabulary from synthetic training data
- Handle speech artifacts: "um", "uh", "so basically", "you know"
- Lowercase everything (spoken text has no case)
- Add special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- Output: `List[int]` of token IDs, padded/truncated to `MAX_SEQ_LEN`

**Interface:**

```python
class Tokenizer:
    def __init__(self, vocab_size: int = 30_000):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
    
    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from a list of texts."""
        ...
    
    def encode(self, text: str) -> List[int]:
        """Tokenize text to list of token IDs (padded to MAX_SEQ_LEN)."""
        ...
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        ...
    
    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to file."""
        ...
    
    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from file."""
        ...
```

**Tokenization rules:**
1. Lowercase
2. Split on whitespace and punctuation
3. Remove filler words if desired (optional, configurable)
4. Map words to IDs using word2idx
5. Unknown words → `[UNK]` token
6. Pad or truncate to `MAX_SEQ_LEN`

---

### 3. `brain/backbone.py` — Shared Transformer Backbone

**Purpose:** The core encoder that processes tokenized text into contextual representations.

**Architecture:**
```
Input: token_ids [batch, seq_len]
  │
  ▼
Embedding(vocab_size → EMBEDDING_DIM)  +  PositionalEncoding(max_len, dim)
  │
  ▼
TransformerEncoderBlock × NUM_LAYERS:
  MultiHeadAttention(NUM_HEADS, EMBEDDING_DIM)
  LayerNorm + Residual
  FeedForward(EMBEDDING_DIM → FF_DIM → EMBEDDING_DIM)
  LayerNorm + Residual
  │
  ├──→ context [batch, EMBEDDING_DIM]   (CLS token or attention pool)
  └──→ tokens [batch, seq_len, EMBEDDING_DIM]  (all token outputs)
```

**Interface:**

```python
class PositionalEncoding(nn.Module):
    """Learned positional encoding."""
    def __init__(self, max_len: int, d_model: int): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float): ...
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor: ...

class Backbone(nn.Module):
    """Shared backbone for all heads."""
    def __init__(self): ...
    
    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len] integer tensor
        Returns:
            context: [batch, EMBEDDING_DIM] — global representation
            tokens: [batch, seq_len, EMBEDDING_DIM] — per-token representations
        """
        ...
```

**Implementation notes:**
- Use pre-norm (LayerNorm before attention/FF) for training stability
- Global attention pooling: learned query attends to all tokens → single vector
- Support causal mask for future use (action decoder might need it)

---

### 4. `brain/heads.py` — Task Heads

**Purpose:** 10 specialized heads that take backbone output and produce task-specific predictions.

**Each head is a standalone `nn.Module`** that takes either `context` or `tokens` as input.

```python
# Head 1: Content Classifier
class ClassifierHead(nn.Module):
    """Classify note type. Input: context [batch, 256]"""
    def __init__(self, num_types: int = len(NoteType)): ...
    def forward(self, context: torch.Tensor) -> torch.Tensor:  # [batch, num_types]
        ...

# Head 2: Entity Tagger (BIO)
class EntityHead(nn.Module):
    """Per-token BIO tagging. Input: tokens [batch, seq_len, 256]"""
    def __init__(self, num_bio_tags: int = ...): ...  # 2 * len(EntityType) + 1 (O)
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [batch, seq_len, num_tags]
        ...

# Head 3: Priority Predictor
class PriorityHead(nn.Module):
    """Regression 0-1. Input: context [batch, 256]"""
    def __init__(self): ...
    def forward(self, context: torch.Tensor) -> torch.Tensor:  # [batch, 1]
        ...

# Head 4: Urgency Predictor
class UrgencyHead(nn.Module):
    """4-class classification. Input: context [batch, 256]"""
    def __init__(self): ...
    def forward(self, context: torch.Tensor) -> torch.Tensor:  # [batch, 4]
        ...

# Head 5: Sentiment Analyzer
class SentimentHead(nn.Module):
    """5-class classification. Input: context [batch, 256]"""
    def __init__(self): ...
    def forward(self, context: torch.Tensor) -> torch.Tensor:  # [batch, 5]
        ...

# Head 6: Action Extractor (Seq2Seq)
class ActionHead(nn.Module):
    """Generate action text. Input: context [batch, 256]"""
    def __init__(self): ...
    def forward(self, context: torch.Tensor, 
                target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        During training: teacher forcing, return logits [batch, max_len, vocab_size]
        During inference: autoregressive generation, return token_ids [batch, max_len]
        """
        ...

# Head 7: Summary Extractor
class SummaryHead(nn.Module):
    """Per-token scoring for extractive summarization. Input: tokens [batch, seq_len, 256]"""
    def __init__(self): ...
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [batch, seq_len]
        ...

# Head 8: Temporal Resolver
class TemporalHead(nn.Module):
    """Predict temporal properties. Input: context [batch, 256]"""
    def __init__(self): ...
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns: has_date, relative_time, month predictions"""
        ...

# Head 9: Topic Classifier (multi-label)
class TopicHead(nn.Module):
    """Multi-label classification. Input: context [batch, 256]"""
    def __init__(self, num_topics: int = len(TOPICS)): ...
    def forward(self, context: torch.Tensor) -> torch.Tensor:  # [batch, num_topics]
        ...

# Head 10: Relationship Detector
class RelationHead(nn.Module):
    """Classify relationship between two entities. Input: concatenated embeddings"""
    def __init__(self, num_relations: int = len(RELATION_TYPES)): ...
    def forward(self, entity_pair: torch.Tensor) -> torch.Tensor:  # [batch, num_relations]
        ...
```

**BIO Tag count calculation:**
```python
# For entity tagger: O + 2 * number of entity types
NUM_BIO_TAGS = 1 + 2 * len(EntityType)  # = 1 + 2*22 = 45
```

---

### 5. `brain/model.py` — Full Multi-Task Model

**Purpose:** Wire backbone + all heads into a single model. One forward pass produces all outputs.

```python
class BrainModel(nn.Module):
    """Multi-task neural network for text understanding."""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None  # set externally
        self.backbone = Backbone()
        self.classifier = ClassifierHead()
        self.entity_tagger = EntityHead()
        self.priority = PriorityHead()
        self.urgency = UrgencyHead()
        self.sentiment = SentimentHead()
        self.action = ActionHead()
        self.summary = SummaryHead()
        self.temporal = TemporalHead()
        self.topics = TopicHead()
        self.relation = RelationHead()
    
    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            token_ids: [batch, seq_len]
        
        Returns:
            Dict with keys:
              classifier: [batch, num_types]
              entities: [batch, seq_len, num_bio_tags]
              priority: [batch, 1]
              urgency: [batch, 4]
              sentiment: [batch, 5]
              action: [batch, max_len, vocab_size] or [batch, max_len]
              summary_scores: [batch, seq_len]
              temporal: dict of tensors
              topics: [batch, num_topics]
        """
        context, tokens = self.backbone(token_ids)
        
        return {
            "classifier": self.classifier(context),
            "entities": self.entity_tagger(tokens),
            "priority": self.priority(context),
            "urgency": self.urgency(context),
            "sentiment": self.sentiment(context),
            "action": self.action(context),
            "summary_scores": self.summary(tokens),
            "temporal": self.temporal(context),
            "topics": self.topics(context),
        }
    
    def predict(self, text: str) -> "Understanding":
        """
        High-level inference. Tokenize → forward → decode → Understanding.
        
        THIS IS WHAT INSTANCE B CALLS.
        """
        ...
    
    def save(self, path: str) -> None: ...
    
    @classmethod
    def load(cls, path: str) -> "BrainModel": ...
```

---

### 6. `brain/embedder.py` — Sentence Embeddings

**Purpose:** Wrap sentence-transformers for text similarity.

```python
class TextEmbedder:
    """Local sentence embedding model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
    
    def embed(self, text: str) -> List[float]:
        """Embed a single text. Returns 384-dim list."""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts. Returns list of 384-dim lists."""
        ...
    
    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        ...
```

---

### 7. `brain/entity_linker.py` — Entity Resolution

**Purpose:** Link extracted entities to existing entity pages in the vault.

```python
class EntityLinker:
    """Resolve entities to known vault entities."""
    
    def __init__(self, entity_index_path: str = "vault_index/entities.faiss"):
        # Load FAISS index of known entities
        ...
    
    def resolve(self, entity: Entity, threshold: float = 0.85) -> Optional[str]:
        """
        Given an extracted entity, find the best match in the vault.
        Returns vault page name or None.
        
        Example:
          Entity(name="Sarah", type=PERSON) → "sarah-chen"
          Entity(name="VAD", type=TOOL) → "silero-vad"
        """
        ...
    
    def add_entity(self, name: str, entity_type: EntityType, embedding: List[float]) -> None:
        """Add a new entity to the index."""
        ...
    
    def build_index(self, vault_path: str = "vault/") -> None:
        """Scan vault/ and build entity index from existing pages."""
        ...
```

---

### 8. `brain/data.py` — Synthetic Data Generator

**Purpose:** Generate training data from templates.

**This is critical.** The quality of synthetic data determines initial model accuracy.

```python
class SyntheticDataGenerator:
    """Generate labeled training examples from templates."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate(self, num_samples: int = 15_000) -> List[Dict]:
        """
        Generate synthetic training data.
        
        Returns list of dicts:
        {
            "text": "I need to fix the VAD threshold before Friday",
            "type": "task",
            "priority": 0.8,
            "urgency": "soon",
            "sentiment": "neutral",
            "entities": [
                {"name": "VAD", "type": "tool"},
                {"name": "Friday", "type": "date"}
            ],
            "action": "fix VAD threshold",
            "topics": ["technology", "engineering"],
            "has_temporal": True,
        }
        """
        ...
    
    def _generate_tasks(self, n: int) -> List[Dict]: ...
    def _generate_ideas(self, n: int) -> List[Dict]: ...
    def _generate_observations(self, n: int) -> List[Dict]: ...
    def _generate_learnings(self, n: int) -> List[Dict]: ...
    def _generate_decisions(self, n: int) -> List[Dict]: ...
    def _generate_meetings(self, n: int) -> List[Dict]: ...
    def _generate_reminders(self, n: int) -> List[Dict]: ...
    def _generate_person_info(self, n: int) -> List[Dict]: ...
    def _generate_preferences(self, n: int) -> List[Dict]: ...
    def _generate_journals(self, n: int) -> List[Dict]: ...
    # ... one generator per note type
    
    def _inject_speech_artifacts(self, text: str) -> str:
        """Add 'um', 'so basically', 'you know', etc."""
        ...
    
    def _inject_entities(self, template: str, entities: Dict) -> str:
        """Fill entity placeholders in templates."""
        ...
    
    def save(self, data: List[Dict], path: str) -> None: ...
    
    @classmethod
    def load(cls, path: str) -> List[Dict]: ...
```

**Template structure (example):**

```python
TASK_TEMPLATES = [
    "I need to {action} the {thing}",
    "TODO: {action} {thing} before {deadline}",
    "don't forget to {action} {thing}",
    "{thing} needs to be {past_action}",
    "I should {action} {thing} when I get a chance",
    "can you remind me to {action} {thing}",
    "the {thing} needs {action} by {deadline}",
    "I want to {action} the {thing} this {timeframe}",
    "need to get {thing} {past_action} before {deadline}",
    "quick task: {action} {thing}",
]

ACTIONS = [
    "fix", "update", "review", "ship", "test", "deploy",
    "write", "rewrite", "debug", "refactor", "check",
    "build", "install", "configure", "optimize", "measure",
    "document", "clean up", "organize", "schedule", "plan",
]

THINGS = [
    "VAD threshold", "wake word model", "README", "test suite",
    "deployment script", "API endpoint", "database migration",
    "authentication flow", "CI pipeline", "documentation",
    "user interface", "performance benchmarks", "error handling",
]
```

**Entity injection patterns:**

```python
PERSON_NAMES = [
    "Sarah", "Mike", "John", "Emily", "David", "Lisa",
    "Alex", "Jordan", "Chris", "Pat", "Sam", "Kim",
]

TOOLS = [
    "Obsidian", "PyTorch", "ONNX", "MLX", "Python",
    "Docker", "Git", "VS Code", "PostgreSQL", "Redis",
]

PROJECTS = [
    "voiceassist", "second-brain", "marketing Q3", "website redesign",
    "mobile app", "data pipeline", "ML model", "API gateway",
]
```

---

### 9. `brain/train.py` — Training Pipeline

**Purpose:** Train the model on synthetic + real data.

```python
class BrainTrainer:
    """Training loop for the multi-task model."""
    
    def __init__(self, model: BrainModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train one epoch. Returns loss dict per head."""
        ...
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate on validation set."""
        ...
    
    def train(self, train_data: List[Dict], val_data: List[Dict], epochs: int = EPOCHS):
        """Full training loop."""
        ...
    
    def retrain_head(self, head_name: str, data: List[Correction], epochs: int = 5):
        """Retrain a single head (fast, for corrections)."""
        ...
    
    def retrain_all(self, data: List[Correction], epochs: int = 10):
        """Full model retrain with correction data."""
        ...
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict): ...
    def load_checkpoint(self, path: str) -> Tuple[int, Dict]: ...
```

**Loss function:**

```python
def compute_loss(outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
    """
    Multi-task loss.
    
    Returns dict of per-head losses + total weighted loss.
    """
    losses = {}
    
    # Classification: CrossEntropy
    losses["classifier"] = F.cross_entropy(
        outputs["classifier"], targets["type_labels"]
    )
    
    # Entity tagging: CrossEntropy per token
    losses["entities"] = F.cross_entropy(
        outputs["entities"].view(-1, outputs["entities"].size(-1)),
        targets["entity_labels"].view(-1),
        ignore_index=-1  # padding
    )
    
    # Priority: MSE regression
    losses["priority"] = F.mse_loss(
        outputs["priority"].squeeze(), targets["priority_scores"]
    )
    
    # Urgency: CrossEntropy
    losses["urgency"] = F.cross_entropy(
        outputs["urgency"], targets["urgency_labels"]
    )
    
    # Sentiment: CrossEntropy
    losses["sentiment"] = F.cross_entropy(
        outputs["sentiment"], targets["sentiment_labels"]
    )
    
    # Action: CrossEntropy on decoder output (teacher forcing)
    losses["action"] = F.cross_entropy(
        outputs["action"].view(-1, outputs["action"].size(-1)),
        targets["action_ids"].view(-1),
        ignore_index=0  # padding
    )
    
    # Summary: Binary cross-entropy per token
    losses["summary"] = F.binary_cross_entropy(
        outputs["summary_scores"], targets["summary_labels"].float()
    )
    
    # Temporal: Multi-task (has_date + relative + month)
    losses["temporal_has_date"] = F.binary_cross_entropy(
        outputs["temporal"]["has_date"], targets["has_date"].float()
    )
    losses["temporal_relative"] = F.cross_entropy(
        outputs["temporal"]["relative"], targets["relative_labels"]
    )
    
    # Topics: Multi-label binary cross-entropy
    losses["topics"] = F.binary_cross_entropy(
        outputs["topics"], targets["topic_labels"].float()
    )
    
    # Weighted total
    total = (
        CLASSIFIER_LOSS_WEIGHT * losses["classifier"] +
        ENTITY_LOSS_WEIGHT * losses["entities"] +
        PRIORITY_LOSS_WEIGHT * losses["priority"] +
        1.0 * losses["urgency"] +
        1.0 * losses["sentiment"] +
        1.0 * losses["action"] +
        0.5 * losses["summary"] +
        1.0 * losses["temporal_has_date"] +
        1.0 * losses["temporal_relative"] +
        TOPIC_LOSS_WEIGHT * losses["topics"]
    )
    losses["total"] = total
    
    return losses
```

---

### 10. `brain/understand.py` — Main Inference API

**Purpose:** This is the entry point that Instance B calls. It wraps everything into a clean API.

```python
class BrainEngine:
    """
    The main brain interface.
    
    Usage:
        engine = BrainEngine.load("checkpoints/brain/latest")
        understanding = engine.understand("I need to fix the VAD threshold by Friday")
        print(understanding.type)  # NoteType.TASK
        print(understanding.priority)  # 0.82
    """
    
    def __init__(self, model: BrainModel, tokenizer: Tokenizer, 
                 embedder: TextEmbedder, entity_linker: EntityLinker):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.entity_linker = entity_linker
    
    def understand(self, text: str, context: Optional[List[Dict]] = None) -> Understanding:
        """
        Full understanding pipeline.
        
        1. Tokenize text
        2. Run through model (all heads)
        3. Decode predictions
        4. Generate embedding
        5. Link entities to vault
        6. Return Understanding object
        """
        ...
    
    def understand_batch(self, texts: List[str]) -> List[Understanding]:
        """Batch inference for efficiency."""
        ...
    
    def should_call_llm(self, understanding: Understanding) -> bool:
        """
        Decide if the LLM should handle this instead.
        
        Returns True if:
          - confidence < 0.70
          - type is ambiguous
          - complex temporal reasoning needed
        """
        ...
    
    def suggest_links(self, understanding: Understanding, 
                      candidate_notes: List[Dict]) -> List[str]:
        """Suggest related notes based on embedding similarity."""
        ...
    
    @classmethod
    def load(cls, checkpoint_dir: str) -> "BrainEngine":
        """Load a trained brain engine."""
        ...
    
    def save(self, checkpoint_dir: str) -> None:
        """Save brain engine state."""
        ...
```

---

### 11. `brain/correct.py` — Correction Logging

**Purpose:** Log user corrections for retraining.

```python
class CorrectionLogger:
    """Manages the correction buffer for continuous learning."""
    
    def __init__(self, buffer_path: str = "checkpoints/brain/corrections.jsonl"):
        self.buffer_path = buffer_path
        self.buffer: List[Correction] = []
        self._load_buffer()
    
    def log(self, correction: Correction) -> None:
        """Add a correction to the buffer."""
        ...
    
    def get_training_data(self, min_corrections: int = 20) -> List[Dict]:
        """Get accumulated corrections as training data."""
        ...
    
    def should_retrain(self) -> bool:
        """Check if enough corrections have accumulated."""
        ...
    
    def mark_applied(self, indices: List[int]) -> None:
        """Mark corrections as used for retraining."""
        ...
    
    def stats(self) -> Dict[str, int]:
        """Get correction statistics."""
        ...
    
    def _load_buffer(self) -> None: ...
    def _save_buffer(self) -> None: ...
```

---

## Testing

Every module must have tests. Run with:

```bash
python -m pytest brain/tests/ -v
```

### Test requirements:

| Module | What to test |
|---|---|
| `tokenizer.py` | Encode/decode roundtrip, unknown words, padding |
| `backbone.py` | Output shapes, gradient flow, no NaN |
| `heads.py` | Each head output shape, loss computation |
| `model.py` | Full forward pass, save/load |
| `embedder.py` | Embedding dimension, similarity ordering |
| `data.py` | Generate N samples, all types covered, entity injection |
| `train.py` | Loss decreases over epochs, checkpoint save/load |
| `understand.py` | End-to-end: text → Understanding, all fields populated |
| `correct.py` | Log correction, get training data, retrain trigger |

### Quick smoke test:

```bash
python -c "
from brain.understand import BrainEngine
engine = BrainEngine.load('checkpoints/brain/latest')
u = engine.understand('I need to fix the VAD threshold by Friday')
print(f'Type: {u.type}')
print(f'Priority: {u.priority}')
print(f'Entities: {[(e.name, e.entity_type) for e in u.entities]}')
print(f'Action: {u.action}')
print(f'Topics: {u.topics}')
"
```

---

## What You Output

When Instance B calls `engine.understand(text)`, they get:

```python
Understanding(
    text="I need to fix the VAD threshold by Friday",
    timestamp="2026-08-23T14:30:00Z",
    type=NoteType.TASK,
    priority=0.82,
    urgency=Urgency.SOON,
    sentiment=Sentiment.NEUTRAL,
    confidence=0.87,
    entities=[
        Entity(name="VAD", entity_type=EntityType.TOOL, confidence=0.95),
        Entity(name="Friday", entity_type=EntityType.DATE, confidence=0.99),
    ],
    action="fix VAD threshold",
    summary="Fix VAD threshold by Friday",
    topics=["technology", "engineering"],
    tags=["bug", "vad", "audio"],
    embedding=[0.23, -0.11, 0.45, ...],  # 384 floats
    temporal=TemporalInfo(has_date=True, absolute_date="2026-08-29", ...),
    related_notes=[],
    metadata={},
)
```

That's what you produce. Instance B takes it from there.
