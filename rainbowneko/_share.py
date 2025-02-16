# for training loggers
loggers = None

# for multi-gpu training
local_rank = -1
world_size = 1
device = 'cuda'

# for model related methods
model_callbacks = []

def register_model_callback(callback):
    model_callbacks.append(callback)
    return callback