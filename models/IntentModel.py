import sys
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

sys.path.append('../chatbot_kim')
from config.GlobalParams import MAX_SEQ_LEN

# 의도 분류 모델 모듈
class IntentModel:
    def __init__(self, model_name, proprocess, intent_level):

        # 의도 클래스 별 레이블
        if intent_level == 1:
            self.labels = {0: "일상", 1: "부정", 2: "긍정"}
        elif intent_level == 2:
            self.labels = {0: "중립", 1: "행복", 2: "슬픔", 3: "놀람", 4: "분노", 5: "공포", 6: "혐오"}
        else:
            print('Wrong intent_level. Enter 1 or 2.')

        # 의도 분류 모델 불러오기
        self.model = load_model(model_name)

        # 챗봇 Preprocess 객체
        self.p = proprocess


    # 의도 클래스 예측
    def predict_class(self, query, intent_level):
        # 형태소 분석
        pos = self.p.pos(query)

        # 문장내 키워드 추출(불용어 제거)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        # 패딩처리
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN[intent_level], padding='post')

        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]
