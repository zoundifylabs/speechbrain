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
