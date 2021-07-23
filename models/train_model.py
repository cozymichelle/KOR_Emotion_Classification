# 필요한 모듈 임포트
import sys, getopt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

sys.path.append('../chatbot_kim')
from utils.Preprocess import Preprocess
from config.GlobalParams import MAX_SEQ_LEN

def read_data(intent_level):

    # Read data for selected intent_level
    if intent_level == 1:
        train_file = "data/ChatbotData.csv"
    elif intent_level == 2:
        train_file = "data/ConverseData.csv"
    else:
        print('Wrong intent_level. Enter 1 or 2.')
        sys.exit(2)

    data = pd.read_csv(train_file, delimiter=',')
    queries = data['query'].tolist()
    intents = data['intent'].tolist()

    p = Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin',
            userdic='utils/user_dic.tsv')

    # 단어 시퀀스 생성
    sequences = []
    for sentence in queries:
        pos = p.pos(sentence)
        keywords = p.get_keywords(pos, without_tag=True)
        seq = p.get_wordidx_sequence(keywords)
        sequences.append(seq)
    
    # 단어 인덱스 시퀀스 벡터 ○2
    # 단어 시퀀스 벡터 크기
    padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN[intent_level], padding='post')

    # 학습용, 검증용, 테스트용 데이터셋 생성 ○3
    # 학습셋:검증셋:테스트셋 = 7:2:1
    ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
    ds = ds.shuffle(len(queries))

    return ds, len(padded_seqs), len(p.word_index)


def split_data(ds, data_size):
    train_size = int(data_size * 0.7)
    val_size = int(data_size * 0.2)
    test_size = int(data_size * 0.1)
    
    train_ds = ds.take(train_size).batch(20)
    val_ds = ds.skip(train_size).take(val_size).batch(20)
    test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

    return train_ds, val_ds, test_ds


def create_model(vocab_size, intent_level):
    # Hyperparameters
    dropout_prob = 0.5
    EMB_SIZE = 128
    VOCAB_SIZE = vocab_size + 1 #전체 단어 개수
    label_size = 3 if intent_level==1 else 7
    print(label_size)

    # CNN 모델 정의  ○4
    input_layer = Input(shape=(MAX_SEQ_LEN[intent_level],))
    embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN[intent_level])(input_layer)
    dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

    conv1 = Conv1D(
        filters=128,
        kernel_size=3,
        padding='valid',
        activation=tf.nn.relu)(dropout_emb)
    pool1 = GlobalMaxPool1D()(conv1)
    
    conv2 = Conv1D(
        filters=128,
        kernel_size=4,
        padding='valid',
        activation=tf.nn.relu)(dropout_emb)
    pool2 = GlobalMaxPool1D()(conv2)
    
    conv3 = Conv1D(
        filters=128,
        kernel_size=5,
        padding='valid',
        activation=tf.nn.relu)(dropout_emb)
    pool3 = GlobalMaxPool1D()(conv3)
    
    # 3,4,5gram 이후 합치기
    concat = concatenate([pool1, pool2, pool3])
    
    hidden = Dense(128, activation=tf.nn.relu)(concat)
    dropout_hidden = Dropout(rate=dropout_prob)(hidden)
    logits = Dense(label_size, name='logits')(dropout_hidden)
    predictions = Dense(label_size, activation=tf.nn.softmax)(logits)

    # 모델 생성  ○5
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) 

    return model


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["intentLevel="])
    except getopt.GetoptError:
        print('train_model_kim.py -i <intentLevel>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train_model_kim.py -i <intentLevel>')
            sys.exit()
        elif opt in ("-i", "--intentLevel"):
            intent_level = int(arg)

    # Read training data
    ds, data_size, vocab_size = read_data(intent_level)

    # Split data into training, validationa and test sets
    train_ds, val_ds, test_ds = split_data(ds, data_size)

    # Define the training model
    model = create_model(vocab_size, intent_level)

    # 모델 학습 ○6
    EPOCH = 5 if intent_level==1 else 40
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)
    
    # 모델 평가(테스트 데이터 셋 이용) ○7
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))
    print('loss: %f' % (loss))

    # 모델 저장  ○8
    model_name = 'models/intent_model_' + str(intent_level) + '.h5'
    model.save(model_name)


if __name__ == "__main__":
    main(sys.argv[1:])