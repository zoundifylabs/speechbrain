# Create environment
    
    conda create --name <env_name> python=3.9
    conda activate <env_name>

# Download Repository
    
    git clone https://github.com/zoundifylabs/speechbrain.git
    cd speechbrain
    pip install -r requirements.txt
    pip install --editable .


# Test Installation
    pytest tests
    pytest --doctest-modules speechbrain

# Perform Emotion recognition
    from speechbrain.inference.interfaces import foreign_class
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py",                          classname="CustomEncoderWav2vec2Classifier")
    out_prob, score, index, text_lab = classifier.classify_file("path/to/audiofile(.wav)")
    print(text_lab)


# staging-t4 setup
    cd /home/ubuntu/speechbrain
    conda activate speechbrain
    python emotion_test.py

    The following message will come:
    
    INFO:     Will watch for changes in these directories: ['/home/ubuntu/speechbrain']
    INFO:     Uvicorn running on http://0.0.0.0:7902 (Press CTRL+C to quit)
    INFO:     Started reloader process [8783] using WatchFiles

    check docs at http://0.0.0.0:7902/docs
    curl -X 'GET' \
      'http://0.0.0.0:7902/emotion?wav_path=%2Fhome%2Fubuntu%2Fzound%2FPROCESSING%2FI8t9nHhUrDQ%2Foutput%2FHindi%2Faudio%2Fsentence_10.mp3' \
      -H 'accept: application/json'

    http://0.0.0.0:7902/emotion?wav_path=%2Fhome%2Fubuntu%2Fzound%2FPROCESSING%2FI8t9nHhUrDQ%2Foutput%2FHindi%2Faudio%2Fsentence_10.mp3
