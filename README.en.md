
# A Simple n-gram Model for Emotion Classification
Trained with Korean texts, this simple CNN-based model classifies emotions.
This code is based on the codes from ["처음 배우는 딥러닝 챗봇"](https://github.com/keiraydev/chatbot)

# Models
* Model 1 trained with [Chatbot_data](https://github.com/songys/Chatbot_data)
* Model 2 trained with [한국어 감정 정보가 포함된 단발성 대화 데이터셋](https://aihub.or.kr/keti_data_board/language_intelligence)

# Setup
## Install Libraries and Dependencies
Clone library on github and install requirements.
```
git clone https://github.com/cozyimchelle/KOR_Emotion_Classification
cd KOR_Emotion_Classification
export PYTHONPATH=.
pip3 install -r requirements.txt
```

## Test models
Enter interactive test mode by running `test.py`
```
python test.py
```

## Train models
* To train Model 1, download the data from [Chatbot_data](https://github.com/songys/Chatbot_data).
Store the data in `data/` folder as `ChatbotData.csv`.
Run the following script:
```
python models/train_model.py -i 1
```

* To train Model 2, download the data from [한국어 감정 정보가 포함된 단발성 대화 데이터셋](https://aihub.or.kr/keti_data_board/language_intelligence).
Store the data in `data/` folder as `ConverseData.csv`.
Run the following script:
```
python models/train_model.py -i 2
```