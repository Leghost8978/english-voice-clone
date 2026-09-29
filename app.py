import os
import tempfile
import gradio as gr
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
model = None


def load_model():
    global model
    if model is None:
        model = TTS(
            model_name=MODEL_NAME,
            progress_bar=True,
            gpu=False
        )
    return model


def generate_audio(text, speaker_audio):
    if not text or not text.strip():
        raise gr.Error("请先输入英语课文。")

    if not speaker_audio:
        raise gr.Error("请先上传自己的声音录音。")

    if len(text) > 1500:
        raise gr.Error("本次最多输入1500个字符，请分段生成。")

    try:
        tts = load_model()
        output_path = os.path.join(
            tempfile.gettempdir(),
            "english_reading.wav"
        )

        tts.tts_to_file(
            text=text.strip(),
            speaker_wav=speaker_audio,
            language="en",
            file_path=output_path,
            split_sentences=True
        )

        return output_path

    except Exception as e:
        raise gr.Error(
            "生成失败，请检查模型运行状态和录音格式。"
            + str(e)[:300]
        )


with gr.Blocks(title="我的英语声音克隆器") as demo:
    gr.Markdown(
        """
        # 🎧 我的英语声音克隆器

        上传自己的录音，粘贴英语课文，
        尝试用自己的音色生成朗读音频。

        本应用使用开源 XTTS-v2 模型。
        首次运行需要下载模型，CPU生成可能较慢。
        """
    )

    speaker_audio = gr.Audio(
        label="上传自己的声音录音",
        type="filepath",
        sources=["upload", "microphone"]
    )

    text = gr.Textbox(
        label="英语课文",
        placeholder="Paste your English passage here...",
        lines=8
    )

    generate_button = gr.Button(
        "生成英语朗读",
        variant="primary"
    )

    output_audio = gr.Audio(
        label="生成的朗读音频",
        type="filepath"
    )

    generate_button.click(
        fn=generate_audio,
        inputs=[text, speaker_audio],
        outputs=output_audio
    )


demo.launch()
