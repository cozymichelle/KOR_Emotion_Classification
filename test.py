from utils.Preprocess import Preprocess
from models.IntentModel import IntentModel

p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin',
               userdic='utils/user_dic.tsv')

intent_level = input("원하는 감정 분류 모델 선택('1' 또는 '2' 입력): ")

model_name = 'models/intent_model_' + intent_level + '.h5'
intent = IntentModel(model_name=model_name, proprocess=p, intent_level = int(intent_level))

print("선택한 모델의 감정 레이블 : ", intent.labels)

while True:
    query = input("문장 입력 ('1' 입력 시, 종료): ")

    if query == '1':
        break

    c = intent.predict_class(query)
    print(intent.labels[c])