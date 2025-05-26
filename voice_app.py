import streamlit as st
import torch
import librosa
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from fuzzywuzzy import process

# Загрузка модели
@st.cache_resource(show_spinner=True)
def load_asr_model():
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
    model.eval()
    return processor, model

processor, model = load_asr_model()

commands = ["лево", "право", "вверх", "вниз", "стоп", 'верх']
st.title("🎮 2D Голосовое управление персонажем")

# Позиции
if "x" not in st.session_state:
    st.session_state.x = 50
if "y" not in st.session_state:
    st.session_state.y = 50
if "gif_url" not in st.session_state:
    st.session_state.gif_url = None

# Гифки
IDLE_GIF = 'https://gifs.obs.ru-moscow-1.hc.sbercloud.ru/387209d24721c82f8a2da74338353e39df081271a162e904e75d79ca3efb9e0e.gif'
WALK_LEFT_GIF = 'https://gifs.obs.ru-moscow-1.hc.sbercloud.ru/387209d24721c82f8a2da74338353e39df081271a162e904e75d79ca3efb9e0e.gif'
WALK_RIGHT_GIF = 'https://gifs.obs.ru-moscow-1.hc.sbercloud.ru/387209d24721c82f8a2da74338353e39df081271a162e904e75d79ca3efb9e0e.gif'
WALK_UP_GIF = 'https://gifs.obs.ru-moscow-1.hc.sbercloud.ru/387209d24721c82f8a2da74338353e39df081271a162e904e75d79ca3efb9e0e.gif'
WALK_DOWN_GIF = 'https://gifs.obs.ru-moscow-1.hc.sbercloud.ru/387209d24721c82f8a2da74338353e39df081271a162e904e75d79ca3efb9e0e.gif'
STOP_GIF = 'https://gifs.obs.ru-moscow-1.hc.sbercloud.ru/387209d24721c82f8a2da74338353e39df081271a162e904e75d79ca3efb9e0e.gif'

# Обработка команды
def move_character(command):
    if command == "лево":
        st.session_state.x = max(0, st.session_state.x - 25)
        st.session_state.gif_url = WALK_LEFT_GIF
    elif command == "право":
        st.session_state.x = min(90, st.session_state.x + 25)
        st.session_state.gif_url = WALK_RIGHT_GIF
    elif command in ["вверх", "верх"]:
        st.session_state.y = max(0, st.session_state.y - 25)  # ДВИЖЕНИЕ ВВЕРХ = уменьшение Y
        st.session_state.gif_url = WALK_UP_GIF
    elif command == "вниз":
        st.session_state.y = min(90, st.session_state.y + 25)  # ДВИЖЕНИЕ ВНИЗ = увеличение Y
        st.session_state.gif_url = WALK_DOWN_GIF
    elif command == "стоп":
        st.session_state.gif_url = STOP_GIF
    else:
        st.session_state.gif_url = IDLE_GIF

# Распознавание
def transcribe_audio(audio_path):
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0]).lower().strip()
        st.write(f"Распознанный текст: **{transcription}**")
        for cmd in commands:
            if cmd in transcription:
                return cmd
        return None
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None

def transcribe_audio(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0]).lower().strip()
    
    # Нечёткое сравнение
    best_match = process.extractOne(transcription, commands, score_cutoff=70)
    return best_match[0] if best_match else None

# Интерфейс
tab1, tab2 = st.tabs(["📁 Загрузить", "🎙 Запись"])

with tab1:
    audio_file = st.file_uploader("Загрузите WAV", type=["wav"])
    if audio_file and st.button("Распознать"):
        st.audio(audio_file)
        command = transcribe_audio(audio_file)
        if command:
            move_character(command)
            st.success(f"Команда: **{command}**")

with tab2:
    if st.button("Записать 2 сек"):
        fs = 16000
        recording = sd.rec(int(2 * fs), samplerate=fs, channels=1)
        sd.wait()
        write("temp.wav", fs, recording)
        st.audio("temp.wav")
        command = transcribe_audio("temp.wav")
        if command:
            move_character(command)
            st.success(f"Команда: **{command}**")

st.markdown(f"""
    <div style="
        position: relative;
        width: 100%;
        height: 500px;
        background-image: url('https://i.pinimg.com/originals/f9/49/c8/f949c8139770a539f056a8383ba04825.png');
        background-size: cover;
        background-position: center;
        border: 2px solid #999;
        border-radius: 10px;
        overflow: hidden;
    ">
        <img src="{st.session_state.gif_url or IDLE_GIF}"
             style="position: absolute;
                    left: {st.session_state.x}%;
                    top: {st.session_state.y}%;
                    width: 100px;">
    </div>
""", unsafe_allow_html=True)

st.markdown("""
### 📋 Инструкция по использованию:

1. **Выберите способ ввода команды**:
   - Вкладка "📁 Загрузить" - для загрузки аудиофайла (формат WAV)
   - Вкладка "🎙 Запись" - для записи голоса через микрофон (2 секунды)

2. **Доступные голосовые команды**:
   - `лево` - движение влево
   - `право` - движение вправо
   - `вверх`/`верх` - движение вверх
   - `вниз` - движение вниз
   - `стоп` - остановка

3. **После распознавания**:
   - Персонаж переместится согласно команде
   - Вы увидите распознанный текст и выполненную команду
""")