from .stgnn import STGNN, ProposedSTGNN, ModifiedSTGNN
from .stgnn import ProposedNoSkipConnectionSTGNN

__all__ = ['STGNN',
           'ProposedSTGNN',
           'ModifiedSTGNN',
           'ProposedNoSkipConnectionSTGNN']

# Optional TensorFlow-based models.
# Keep STGNN importable even when tensorflow is not installed.
try:
  from .lstm_model import create_lstm_model, create_stateless_lstm_model
  from .seq2seq import Encoder, Decoder, DecoderWithAttention
  __all__.extend(['create_lstm_model',
                  'create_stateless_lstm_model',
                  'Encoder',
                  'Decoder',
                  'DecoderWithAttention'])
except ModuleNotFoundError:
  pass