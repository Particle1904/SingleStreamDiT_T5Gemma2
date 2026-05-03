from config import Config

def parse_run_name(learning_rate=None):
    hs = f"hs_{Config.hidden_size}"
    nh = f"nh_{Config.num_heads}"
    d = f"d_{Config.depth}"
    rd = f"rd_{Config.refiner_depth}"
    ps = f"ps_{Config.patch_size}"
    lt = f"lt_{Config.loss_type}"
    lr = f"lr_{learning_rate}" if learning_rate is not None else f"lr_{Config.learning_rate}"
    sv = f"sv_{Config.shift_val}"
    te = f"te_{get_text_encoder_name()}"
    run_name = f"{hs}-{nh}-{d}-{rd}-{ps}-{lt}-{lr}-{sv}-{te}"
    return run_name

def get_text_encoder_name():
    text_encoder_name = "t5gemma-2-4b"
    if("t5gemma-2-270b" in Config.text_model_id):
        text_encoder_name = "t5gemma-2-270m"
    elif("t5gemma-2-1b" in Config.text_model_id):
        text_encoder_name = "t5gemma-2-1b"
    elif("t5gemma-2-4b" in Config.text_model_id):
        text_encoder_name = "t5gemma-2-4b"
    else:
        text_encoder_name = Config.text_model_id.split("/")[-1]
        
    return text_encoder_name