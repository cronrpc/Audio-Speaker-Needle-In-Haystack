import tarfile
import glob
import os
from huggingface_hub import hf_hub_download


def download_audios():
    wav_files = sorted(glob.glob(os.path.join("audios", '*.wav')))
    if len(wav_files) == 0:
        audios_targz_path = hf_hub_download(repo_id="omniway/Audio_speaker_needle_in_haystack", filename="audios.tar.gz", repo_type="dataset")
        tar = tarfile.open(audios_targz_path, 'r:gz')
        tar.extractall(path='.')
        tar.close()

if __name__ == '__main__':
    download_audios()
