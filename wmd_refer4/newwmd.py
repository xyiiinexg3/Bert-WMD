

def word_cut(sentence):
    LTP_DATA_DIR = 'C:\\Users\\d84105613\\ltp_data'
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    segmentor = Segmentor()  # 初始化实例
    segmentor.load_with_lexicon(cws_model_path, 'lexicon')  # 加载模型
    words = segmentor.segment(sentence)
    segmentor.release()
    words = list(words)
    print(len(set(words)))
    words = [c for c in words if c not in punctuation]
    return ' '.join(words)

def word2vecTrain(text):
    model = Word2Vec(LineSentence(text), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save('Word2VecModel.m')
    model.wv.save_word2vec_format('Word2VecModel.vector', binary=False)