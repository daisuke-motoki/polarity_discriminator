import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from polarity_discriminator.discriminator import PolarityDiscriminator
from polarity_discriminator.jp_mecab import JapaneseMecabWordExtractor


def preprocess_inputs(texts, vectorizer, null_id, unknown_id, max_length=None):
    print("Preprocessing inputs.")
    sequences = list()
    masks = list()
    for text in texts:
        words = vectorizer.tokenizer(text)
        maxl = max_length if max_length else len(words)
        ids = np.ones(maxl, dtype="int")*null_id
        offset = 0 if len(words) > maxl else maxl - len(words)
        mask = np.zeros(maxl, dtype="int")
        for i, word in enumerate(words):
            if i >= maxl:
                break
            ids[i+offset] = vectorizer.vocabulary_.get(word, unknown_id)
            mask[i+offset] = 1
        sequences.append(ids)
        masks.append(mask)
    return sequences, masks


def preprocess_gt(Ys, masks):
    print("Preprocessing ground truth.")
    gts = list()
    for Y, mask in zip(Ys, masks):
        gt = mask*Y/mask.sum()
        gts.append(gt)

    return gts


def get_vectorizer(texts, null_set=(0, ""), unknown_set=(1, "###")):
    print("Making vectorizer.")
    tf_vectorizer = CountVectorizer(max_df=1.0,
                                    min_df=10,
                                    max_features=10000,
                                    stop_words=[null_set[1], unknown_set[1]])
    tf_vectorizer.tokenizer = JapaneseMecabWordExtractor(split_mode="unigram",
                                                         use_all=True)
    tf_vectorizer.fit(texts)
    max_id = max(tf_vectorizer.vocabulary_.values())
    prev_char = tf_vectorizer.get_feature_names()[null_set[0]]
    tf_vectorizer.vocabulary_[null_set[1]] = null_set[0]
    tf_vectorizer.vocabulary_[prev_char] = max_id + 1
    prev_char = tf_vectorizer.get_feature_names()[unknown_set[0]]
    tf_vectorizer.vocabulary_[unknown_set[1]] = unknown_set[0]
    tf_vectorizer.vocabulary_[prev_char] = max_id + 2
    return tf_vectorizer


if __name__ == "__main__":
    # load inputs
    headers = ["resource", "rating", "content"]
    filename = "../data/review_data_jalan.csv"
    vectorizer_file = "vectorizer_jalan.pkl"
    # vectorizer_file = None

    df = pd.read_csv(filename, names=headers)
    # df = df[:10000]
    indexes = np.arange(len(df))
    np.random.seed(0)
    np.random.shuffle(indexes)
    train_last = int(len(df)*0.8)
    train_indexes = indexes[:train_last]
    val_indexes = indexes[train_last:]

    # vectorizer
    if vectorizer_file is None:
        vectorizer = get_vectorizer(df.content.tolist())
        tokenizer = vectorizer.tokenizer
        vectorizer.tokenizer = None
        joblib.dump(vectorizer, "vectorizer.pkl")
        vectorizer.tokenizer = tokenizer
    else:
        vectorizer = joblib.load(vectorizer_file)
        vectorizer.tokenizer = JapaneseMecabWordExtractor(split_mode="unigram",
                                                          use_all=True)
    max_id = max(vectorizer.vocabulary_.values()) + 1
    null_id = vectorizer.vocabulary_[""]
    unknown_id = vectorizer.vocabulary_["###"]
    train_X, train_mask = preprocess_inputs(df.content[train_indexes].tolist(),
                                            vectorizer,
                                            null_id=null_id,
                                            unknown_id=unknown_id,
                                            max_length=100)
    val_X, val_mask = preprocess_inputs(df.content[val_indexes].tolist(),
                                        vectorizer,
                                        null_id=null_id,
                                        unknown_id=unknown_id,
                                        max_length=100)
    train_Y = preprocess_gt(df.rating[train_indexes].tolist(),
                            train_mask)
    val_Y = preprocess_gt(df.rating[val_indexes].tolist(),
                          val_mask)

    # create network
    network_architecture = dict(
        max_sequence_len=100,
        # max_sequence_len=None,
        n_word=max_id,
        word_dim=300,
        n_lstm_unit1=100,
        rate_lstm_drop=0.2,
    )
    discriminator = PolarityDiscriminator()
    discriminator.build(network_architecture)

    checkpoints = "./checkpoints/weights.{epoch:02d}-{val_loss:.4f}.hdf5"
    discriminator.train(np.array(train_X),
                        np.array(train_Y),
                        np.array(val_X),
                        np.array(val_Y),
                        epochs=10,
                        batch_size=32,
                        learning_rate=1e-3,
                        checkpoints=checkpoints,
                        shuffle=True)
