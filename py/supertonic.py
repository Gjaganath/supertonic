from helper import load_text_to_speech, load_voice_style
import sounddevice as sd
import soundfile as sf
import os

class TTSEngine:
    def __init__(
            self,
            total_step: int=5,
            speed: float=1.05,
            n_test: int=4,
            save_dir: str="results",
            voice_style_paths: str="..assets/voice_styles/M1.json",
            use_gpu: bool=False,
            batch: bool=False
    ):
    
        self.total_step = total_step
        self.speed = speed
        self.n_test = n_test
        self.save_dir = save_dir
        self.voice_style_paths = voice_style_paths
        self.use_gpu = use_gpu
        self.batch = batch
        self.onnx_dir = "..assets/onnx"

    def initiate(self):
        self.text_to_speech = load_text_to_speech(self.onnx_dir, self.use_gpu)
        self.style = load_voice_style([self.voice_style_paths], verbose=False)

    def synthesize(self, text: str, output: str) -> None:
        for n in range(self.n_test):
            if self.batch:
                wav, duration = self.text_to_speech.batch([text], self.style, self.total_step, self.speed)
            else:
                wav, duration = self.text_to_speech(text, self.style, self.total_step, self.speed)

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            fname = output
            w = wav[0, : int(self.text_to_speech.sample_rate * duration[0].item())]  # [T_trim]
            sf.write(os.path.join(self.save_dir, fname), w, self.text_to_speech.sample_rate)

    def synthesize_play(self, text: str, block: bool=True) -> None:
        for n in range(self.n_test):
            if self.batch:
                wav, duration = self.text_to_speech.batch([text], self.style, self.total_step, self.speed)
            else:
                wav, duration = self.text_to_speech(text, self.style, self.total_step, self.speed)

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            print(duration[0].item())

            w = wav[0, : int(self.text_to_speech.sample_rate * duration[0].item())]
            sd.play(w, self.text_to_speech.sample_rate)
            if block: sd.wait()


if __name__ == "__main__":
    engine = TTSEngine(speed=1.2, n_test=1)
    engine.initiate()

    engine.synthesize_play("Hi, this is super tonic TTS")