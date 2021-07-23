# 단순 감정 분류 모델
한국어 감정 데이터를 이용하여 문장의 감정을 분류하는 모델입니다.
해당 코드는 ["처음 배우는 딥러닝 챗봇"](https://github.com/keiraydev/chatbot)의 코드를 사용하였습니다.
To read this in English, click [here](README.en.md)

# 모델 종류
* 모델 1: [Chatbot_data](https://github.com/songys/Chatbot_data) 데이터로 학습. 3가지 감정("일상", "부정", "긍정")으로 분류.
* 모델 2: [한국어 감정 정보가 포함된 단발성 대화 데이터셋](https://aihub.or.kr/keti_data_board/language_intelligence) 데이터로 학습. 7가지 감정("중립", "행복", "슬픔", "놀람", "분노", "공포", "혐오")으로 분류.

# 설치 및 실행
## 설치
해당 라이브러리를 복제 후, 필수 라이브러리 설치.
```
git clone https://github.com/cozyimchelle/KOR_Emotion_Classification
cd KOR_Emotion_Classification
export PYTHONPATH=.
pip3 install -r requirements.txt
```

## 모델 테스트
`test.py`를 실행하여 채팅모드로 모델 실험.
```
python test.py
```

## 모델 학습
* 모델 1: [Chatbot_data](https://github.com/songys/Chatbot_data)에서 데이터 다운로드.
`data/` 폴더에 `ChatbotData.csv`로 저장.
다음 스크립트를 입력:
```
python models/train_model.py -i 1
```

* 모델 2: [AI 허브](https://aihub.or.kr/keti_data_board/language_intelligence)에서 "한국어 감정 정보가 포함된 단발성 대화 데이터셋" 데이터 다운로드.
`data/` 폴더에 `ConverseData.csv`로 저장.
다음 스크립트를 입력:
```
python models/train_model.py -i 2
```