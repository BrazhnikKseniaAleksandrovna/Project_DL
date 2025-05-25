import streamlit as st
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import pickle
import io
import numpy as np
import tempfile
from pydub import AudioSegment

st.set_page_config(page_title="Голосовые команды", layout="centered")
st.title("🎤 Распознавание голосовых команд из файла")

SAMPLE_CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

class SpeechCommandModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.gap(x)
        x = self.fc(x)
        return x

# Загрузка модели
device = torch.device('cpu')
model = SpeechCommandModel(num_classes=len(SAMPLE_CLASSES))
model.load_state_dict(torch.load("speech_command_cnn.pth", map_location=device))
model.eval()

# Загрузка словаря меток
with open("label2idx.pkl", "rb") as f:
    label2idx = pickle.load(f)
idx2label = {v: k for k, v in label2idx.items()}

st.markdown("""
#### Инструкция:
1. Запишите команду через [online-voice-recorder.com/ru](https://online-voice-recorder.com/ru) или любое другое средство
2. Сохраните как `.wav` или `.mp3` файл
3. Загрузите его ниже — система сама приведёт формат к нужному
""")

uploaded_file = st.file_uploader("Загрузите аудиофайл (wav/mp3)", type=["wav", "mp3"])
if uploaded_file is not None:
    st.audio(uploaded_file)

    try:
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            if uploaded_file.name.endswith(".mp3"):
                sound = AudioSegment.from_file(uploaded_file, format="mp3")
                sound = sound.set_frame_rate(16000).set_channels(1)
                sound.export(tmp_wav.name, format="wav")
            else:
                tmp_wav.write(uploaded_file.read())

            waveform, sr = torchaudio.load(tmp_wav.name)

        # Моно
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Приведение к 16 кГц
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        # Приведение к 1 секунде
        if waveform.shape[1] < 16000:
            pad_len = 16000 - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :16000]

        # Спектрограмма
        mel = T.MelSpectrogram(sr, n_mels=64)(waveform)
        mel = T.AmplitudeToDB()(mel)
        mel = mel.unsqueeze(0)
        mel = torch.nn.functional.pad(mel, (0, max(0, 128 - mel.shape[-1])))
        mel = mel[:, :, :, :128]

        # Предсказание
        out = model(mel)
        pred = out.argmax(dim=1).item()
        command = idx2label.get(pred, "Неизвестно")

        st.success(f"✅ Распознанная команда: **{command}**")

    except Exception as e:
        st.error(f"❌ Ошибка при обработке: {e}")
