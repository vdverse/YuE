# generate_music.py

def generate_music(
    genre_txt,
    lyrics_txt,
    audio_prompt_path="",
    use_audio_prompt=False,
    use_dual_tracks_prompt=False,
    output_dir="./output",
    keep_intermediate=False,
):
    import os
    import sys
    import random
    import uuid
    import numpy as np
    import torch

    # Fix import paths
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))

    from inference.infer import main as run_inference

    # Prepare args as an object
    class Args:
        stage1_model = "m-a-p/YuE-s1-7B-anneal-en-cot"
        stage2_model = "m-a-p/YuE-s2-1B-general"
        max_new_tokens = 3000
        repetition_penalty = 1.1
        run_n_segments = 2
        stage2_batch_size = 4
        genre_txt = genre_txt
        lyrics_txt = lyrics_txt
        use_audio_prompt = use_audio_prompt
        audio_prompt_path = audio_prompt_path
        prompt_start_time = 0.0
        prompt_end_time = 30.0
        use_dual_tracks_prompt = use_dual_tracks_prompt
        vocal_track_prompt_path = ""
        instrumental_track_prompt_path = ""
        output_dir = output_dir
        keep_intermediate = keep_intermediate
        disable_offload_model = False
        cuda_idx = 0
        seed = 42
        basic_model_config = "./xcodec_mini_infer/final_ckpt/config.yaml"
        resume_path = "./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth"
        config_path = "./xcodec_mini_infer/decoders/config.yaml"
        vocal_decoder_path = "./xcodec_mini_infer/decoders/decoder_131000.pth"
        inst_decoder_path = "./xcodec_mini_infer/decoders/decoder_151000.pth"
        rescale = True

    # Call the main inference logic from infer.py
    run_inference(Args())
