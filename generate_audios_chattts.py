
import torchaudio
import torch
import os
import soundfile as sf

from IPython.display import Audio
from tqdm import tqdm

import ChatTTS.ChatTTS as ChatTTS


"""

Audio Speaker needle in haystack
cronrpc
https://github.com/cronrpc

"""


def generator_by_seed(text, seed):
    texts = text

    torch.manual_seed(seed)

    rand_spk = chat.sample_random_speaker()

    params_infer_code = {
        'spk_emb': rand_spk, # add sampled speaker 
        'temperature': .3, # using custom temperature
        'top_P': 0.7, # top P decode
        'top_K': 20, # top K decode
    }

    params_refine_text = {
        'prompt': '[oral_2][laugh_0][break_6]'
    } 

    wavs = chat.infer(texts, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

    return wavs


def generator_and_save(text, seed, base_dir, index_start=0):
    os.makedirs(base_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(base_dir, f"seed{seed}_index{index_start}.wav")):
        return
    
    wavs = generator_by_seed(text, seed)
    for index, wav in enumerate(wavs, index_start):
        wav = wav.squeeze()
        file_name = f"seed{seed}_index{index}.wav"
        file_path = os.path.join(base_dir, file_name)
        sr = 24000
        sf.write(file_path, wav, sr, 'PCM_24')




if __name__ == '__main__':

    base_dir = "audios"
    MAX_BATCH_SIZE = 4
    seed_start = 0
    seed_step = 1
    seed_number = 1024

    chat = ChatTTS.Chat()
    chat.load_models(compile=False)

    texts = ["明天将有小雨，气温在15到20度之间。",
            "你好！很高兴见到你，今天过得怎么样？", 
            "现在购买，即可享受五折优惠，机会不容错过！",
            "你是真听不懂呀？还是假听不懂呀？",
            "Once upon a time, there was a clever little fox living in a dense forest.",
            "Please open the window to let in some fresh air.",
            "Tomorrow, there will be light rain, with temperatures ranging from 15 to 20 degrees Celsius.",
            "Excuse me, what are some recommended movies currently showing?",
            ]

    for seed in tqdm(range(seed_start, seed_start + seed_number * seed_step, seed_step)):
        num_texts = len(texts)
        num_batches = (num_texts + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

        for batch_idx in range(num_batches):
            index_start = batch_idx * MAX_BATCH_SIZE
            end_idx = min((batch_idx + 1) * MAX_BATCH_SIZE, num_texts)
            batch_texts = texts[index_start:end_idx]
            generator_and_save(batch_texts, seed, base_dir, index_start=index_start)
            generator_and_save(batch_texts, seed, base_dir, index_start=index_start)