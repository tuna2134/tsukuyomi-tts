from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from pathlib import Path
from huggingface_hub import hf_hub_download
from style_bert_vits2.tts_model import TTSModel
import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")


model_file = "model_assets/tyc/tyc_e100_s5300.safetensors"
config_file = "model_assets/tyc/config.json"
style_file = "model_assets/tyc/style_vectors.npy"

for file in [model_file, config_file, style_file]:
    print(file)
    hf_hub_download("tuna2134/tsukuyomi", file, local_dir="model_assets")


assets_root = Path("model_assets")

model = TTSModel(
    model_path=assets_root / model_file,
    config_path=assets_root / config_file,
    style_vec_path=assets_root / style_file,
    device="cpu",
)


app = FastAPI()


@app.get("/")
async def synthesis(text: str) -> StreamingResponse:
    output = io.BytesIO()
    sr, audio = model.infer(text=text)
    write(output, sr, audio)
    output.seek(0)
    return StreamingResponse(output, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
