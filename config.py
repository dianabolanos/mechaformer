class Config:
    # Dataset paths
    DATA_PATH = ""
    
    # Absolute paths to key files
    MECHANISM_WRAPPER_PATH = "./wrapper/mechanism_wrapper.py"
    MECHANISM_CORE_PATH = "./wrapper/mechanism_core.py"
    INFERENCE_MECHANISM_PATH = "./inference_mechanism.py"
    
    # Model architecture
    D_MODEL = 256
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DROPOUT = 0.1
    
    # Data processing
    MAX_SEQ_LEN = 64  # Should match preprocessing default
    COORD_MIN = -10.0  # Minimum coordinate value
    COORD_MAX = 10.0   # Maximum coordinate value
    COORD_BINS = 200   # Number of bins for coordinate discretization
    
    # Vocabulary structure
    NUM_SPECIAL_TOKENS = 4  # <PAD>, <SOS>, <EOS>, <UNK>
    NUM_DSL_TOKENS = 4      # DSL command tokens
    BIN_START_ID = NUM_SPECIAL_TOKENS + NUM_DSL_TOKENS  # 8
    BIN_END_ID = BIN_START_ID + COORD_BINS - 1  # 8 + 2000 - 1 = 2007
    
    # B-spline parameters
    BSPLINE_CONTROL_POINTS = 64  # Fixed number of control points
    BSPLINE_DEGREE = 3  # Cubic B-splines
    BSPLINE_SMOOTHING = 0.0  # No smoothing for exact fitting
    
    # Training
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 30
    GRAD_CLIP = 1.0
    
    # Data loading
    NUM_WORKERS = 2
    DEV_DATA_LIMIT = 1000  # Limit data for development/testing
    
    # Scheduler
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    
    # Save paths
    MODEL_SAVE_PATH = 'mechanism_transformer_weights.pth'
    VOCAB_SAVE_PATH = 'vocab.pkl'
    
    # Device
    DEVICE = 'cuda'  # Will be overridden based on availability 