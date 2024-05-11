from speechbrain.inference.interfaces import foreign_class
import numpy
import uvicorn
from fastapi import FastAPI
from fastapi import Response
app = FastAPI()



classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py",                          classname="CustomEncoderWav2vec2Classifier")


# out_prob, score, index, text_lab = classifier.classify_file("/home/ubuntu/IndicTTS/multilingual-tts/mahaTTS/MahaTTS-main/infer_ref_wavs/f1_5sec_cut/f1_5sec_22k_c1.wav")
# print(out_prob, score, index, text_lab )


@app.get("/emotion")
async def emotion_detection(wav_path:str):
    out_prob, score, index, text_lab = classifier.classify_file(wav_path)
    print(score)
    return {"emotion": text_lab[0], "confidence":int(score.numpy()[0] *100)}

if __name__ == "__main__":
    uvicorn.run("emotion_test:app", host="0.0.0.0", port=7902, reload=True)
