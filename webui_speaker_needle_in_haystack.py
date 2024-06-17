import os
import operator
import glob
import librosa
import argparse
import hashlib
import gradio as gr
import numpy as np
import pickle

from tqdm import tqdm
from modelscope.pipelines import pipeline

"""

Audio Speaker needle in haystack
cronrpc
https://github.com/cronrpc

"""

MAX_DISPLAY_AUDIO_NUMBER = 10
g_gr_audio_list = []


class Speaker_Needle_In_Haystack():
    SAMPLE_RATE = 16000 

    def __init__(self, pickle_support = False) -> None:
        self._load_model()
        self.all_embs = {}
        self.cosine_score = {}
        self.pickle_support = pickle_support
        pass

    def set_audio_list_dir(self, dir_path):
        self.audio_list_dir = dir_path

    def _load_model(self) -> None:
        # could switch model here 

        self.model_name = 'damo/speech_eres2netv2_sv_zh-cn_16k-common'
        self.sv_pipline = pipeline(
            task='speaker-verification',
            model=self.model_name,
            model_revision='v1.0.1'
        )

        # self.model_name = 'iic/speech_campplus_sv_zh-cn_3dspeaker_16k'
        # self.sv_pipline = pipeline(
        #     task='speaker-verification',
        #     model=self.model_name
        # )

    def _get_emb(self, audio) -> None:
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=self.SAMPLE_RATE, mono=True)
            return self.sv_pipline([audio], output_emb=True)['embs'] # (1,196) np array
        elif isinstance(audio, list):
            return self.sv_pipline(audio, output_emb=True)['embs'] # (n,196) np array
        else:
            return self.sv_pipline([audio], output_emb=True)['embs'] # (1,196) np array

    def _cosine_similarity_compute(self, emb1, emb2):
        emb1 = np.squeeze(emb1)
        emb2 = np.squeeze(emb2)
        dot_product = np.dot(emb1, emb2)
        norm_vector1 = np.linalg.norm(emb1)
        norm_vector2 = np.linalg.norm(emb2)
        cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
        return cosine_similarity

    def compute_all_embs(self, batch_size=1):
        wav_files = sorted(glob.glob(os.path.join(self.audio_list_dir, '*.wav')))

        # hash to skip
        file_string = self.model_name + ''.join(wav_files)
        hash_file = hashlib.sha256(file_string.encode()).hexdigest()[:15] + ".pkl"
        if self.pickle_support:
            cache_dir = os.path.join('cache','embs_cache')
            os.makedirs(cache_dir, exist_ok=True)
            hash_file = os.path.join(cache_dir, hash_file)
            if os.path.exists(hash_file):
                print("load pickle embs")
                self.load_all_embs(hash_file)
                return

        self.all_embs = {}
        num_files = len(wav_files)
        num_batches = (num_files + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_files)
            batch_files = wav_files[start_idx:end_idx]
            batch_audio = []

            for file_path in batch_files:
                audio, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, mono=True)
                batch_audio.append(audio)

            embs = self._get_emb(batch_audio)

            for i, file_path in enumerate(batch_files):
                self.all_embs[file_path] = embs[i]

        # save the self.all_embs in hash_value named file
        if self.pickle_support:
            self.save_all_embs(hash_file)

    def compute_target_aduio_cosine_score(self, target_audio):
        self.cosine_score = {}
        target_emb = self._get_emb(target_audio)
        for file_path, emb in self.all_embs.items():
            self.cosine_score[file_path] = self._cosine_similarity_compute(target_emb, emb)

    def get_cosine_next_top_k(self, k, start = 0):
        top_subset = sorted(self.cosine_score.items(), key=operator.itemgetter(1), reverse=True)[start: start + k]
        return top_subset
    
    def save_all_embs(self, hash_file):
        file_path = hash_file
        with open(file_path, 'wb') as file:
            pickle.dump(self.all_embs, file)

    def load_all_embs(self, hash_file):
        file_path = hash_file
        with open(file_path, 'rb') as file:
            self.all_embs = pickle.load(file)


def get_similar_score_audio(audio, start_index):
    output = []
    top_subset = []
    
    if audio != None:
        sr, y = audio
        if len(y.shape) == 2:
            y = np.mean(y, axis=-1)
        audio_16k = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=snih.SAMPLE_RATE)
        snih.compute_target_aduio_cosine_score(audio_16k)
        top_subset = snih.get_cosine_next_top_k(MAX_DISPLAY_AUDIO_NUMBER, start=start_index)

    for i in range(0, len(top_subset)):
        path, score = top_subset[i]
        file_name = os.path.basename(path)
        output.append(
            {
                "__type__":"update",
                "value":path,
                "label":f"{start_index+i}:{file_name} score={score:.4f}"
            }
        )

    for _ in range(0, MAX_DISPLAY_AUDIO_NUMBER - len(top_subset)):
        output.append(
            {
                "__type__":"update",
                "value":None,
                "label":"None"
            }
        )

    return *output, start_index


def get_next_index_zero(audio):
    return get_similar_score_audio(audio, 0)


def get_next_index(audio, start_index):
    return get_similar_score_audio(audio, start_index + 10)


def get_previous_index(audio, start_index):
    return get_similar_score_audio(audio, max(start_index - 10, 0))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Speaker_Needle_In_Haystack demo Launch')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
    parser.add_argument('--server_port', type=int, default=7860, help='Server port')
    parser.add_argument('--batch_size', type=int, default=4, help='the batch_size about embedding generate')
    parser.add_argument('--audio_dir', type=str, default="audios", help='the audio dir which will be compared to target audio')
    parser.add_argument('--disable_pickle_support', action='store_true', help="save emb by pickle")
    args = parser.parse_args()

    pickle_support = not args.disable_pickle_support
    print("pickle support : ", pickle_support)
    snih = Speaker_Needle_In_Haystack(pickle_support=pickle_support)

    snih.set_audio_list_dir(args.audio_dir)
    snih.compute_all_embs(batch_size = args.batch_size)

    with gr.Blocks() as demo:
        gr.Markdown("# 大海捞针 Audio Needle In Haystack")
        with gr.Row():
            audio_input = gr.Audio(
                    label= "Input Audio / 输入音频",
                    visible = True,
                    scale=5,
                    type="numpy",
                    format='wav'
                )
            
            with gr.Column():
                wav_files = sorted(glob.glob(os.path.join("examples", '*.wav')))
                gr.Examples(
                    examples=[
                        *wav_files
                    ],
                    inputs=[
                        audio_input
                    ]
                )
                input_index =  gr.Number(value=0, label="Index")

            btn_get_similar = gr.Button("获取相似音频 Get Similar Score Audio")
            btn_get_previous_index = gr.Button("上一页 Previous Index")
            btn_get_next_index = gr.Button("下一页 Next Index")

        
        gr.Markdown("# 相似音频 similar audio")

        with gr.Column():
            for _ in range(0,MAX_DISPLAY_AUDIO_NUMBER):
                audio_output = gr.Audio(
                    label= "Output Audio",
                    visible = True,
                    scale=5,
                    editable=False
                )
                g_gr_audio_list.append(audio_output)
        
        btn_get_similar.click(
            get_next_index_zero,
            inputs=[
                audio_input
            ],
            outputs=[
                *g_gr_audio_list,
                input_index
            ]
        )

        btn_get_previous_index.click(
            get_previous_index,
            inputs=[
                audio_input,
                input_index
            ],
            outputs=[
                *g_gr_audio_list,
                input_index
            ]
        )

        btn_get_next_index.click(
            get_next_index,
            inputs=[
                audio_input,
                input_index
            ],
            outputs=[
                *g_gr_audio_list,
                input_index
            ]
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port)