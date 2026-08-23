"""
brain.config — Hyperparameters and constants for the brain module.
"""

# =============================================================================
# Model Architecture
# =============================================================================

EMBEDDING_DIM = 256          # backbone hidden dimension
NUM_LAYERS = 4               # transformer layers in backbone
NUM_HEADS = 4                # attention heads per transformer block
FF_DIM = 1024                # feed-forward intermediate dimension
DROPOUT = 0.2
MAX_SEQ_LEN = 512            # maximum input token length

# =============================================================================
# Vocabulary
# =============================================================================

VOCAB_SIZE = 30_000          # tokenizer vocabulary size
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

# Filler words common in spoken transcriptions (optional removal)
FILLER_WORDS = {
    "um", "uh", "erm", "hmm", "like", "you know", "so basically",
    "i mean", "kind of", "sort of", "basically", "actually",
    "literally", "right", "okay so", "so yeah", "anyway",
}

# =============================================================================
# Head Dimensions
# =============================================================================

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

# =============================================================================
# Training
# =============================================================================

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 20
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

# Loss weights per head
CLASSIFIER_LOSS_WEIGHT = 1.0
ENTITY_LOSS_WEIGHT = 2.0     # entities are harder, weight more
PRIORITY_LOSS_WEIGHT = 1.0
URGENCY_LOSS_WEIGHT = 1.0
SENTIMENT_LOSS_WEIGHT = 1.0
ACTION_LOSS_WEIGHT = 1.0
SUMMARY_LOSS_WEIGHT = 0.5
TEMPORAL_LOSS_WEIGHT = 1.0
TOPIC_LOSS_WEIGHT = 0.5      # multi-label, lower weight

# =============================================================================
# Synthetic Data
# =============================================================================

SYNTHETIC_SAMPLES = 15_000
SYNTHETIC_SEED = 42

# =============================================================================
# Checkpoints
# =============================================================================

CHECKPOINT_DIR = "checkpoints/brain"
CHECKPOINT_EVERY = 5         # save every N epochs

# =============================================================================
# Inference
# =============================================================================

LLM_CONFIDENCE_THRESHOLD = 0.70   # below this, route to LLM
HIGH_CONFIDENCE_THRESHOLD = 0.90  # above this, NN handles alone

# =============================================================================
# Integration
# =============================================================================

DEFAULT_VAULT_PATH = "vault/"
DEFAULT_INDEX_PATH = "vault_index/"
DEFAULT_CHECKPOINT_DIR = "checkpoints/brain"

# LLM fallback
LLM_FALLBACK_ENABLED = True
LLM_MIN_CONFIDENCE = 0.50     # below this, ALWAYS use LLM
LLM_MAX_CONFIDENCE = 0.90     # above this, NEVER use LLM

# Correction
CORRECTION_BUFFER_PATH = "checkpoints/brain/corrections.jsonl"
CORRECTION_MIN_FOR_RETRAIN = 20
CORRECTION_HEAD_ONLY = True

# Pipeline
PIPELINE_WRITE_ENABLED = True
PIPELINE_INDEX_ENABLED = True
PIPELINE_LINK_ENABLED = True

# Logging
import os as _os
LOG_LEVEL = _os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
