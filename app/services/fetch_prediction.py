import re
import nltk
import pickle
import warnings
import numpy as np
import pandas as pd
from stop_words import get_stop_words
from nltk.corpus import wordnet as wn
from gensim.models.keyedvectors import KeyedVectors
import gensim
warnings.filterwarnings("ignore")
import os



# creating a list of extra stop-words as these repeatedly appear in all complaints
# xxxx is used in the data to hide sensitive information
stplist = ['title', 'body', 'xxxx']
english_stopwords = get_stop_words(language='english')
english_stopwords += stplist
english_stopwords = list(set(english_stopwords))


def get_wordnet_pos(word):
    """
    Function that determines the the Part-of-speech (POS) tag.
    Acts as input to lemmatizer
    """
    if word.startswith('N'):
        return wn.NOUN
    elif word.startswith('V'):
        return wn.VERB
    elif word.startswith('J'):
        return wn.ADJ
    elif word.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN


def clean_up(text):
    """
    Function to clean data.
    Steps:
    - Removing special characters, numbers
    - Lemmatization
    - Stop-words removal
    - Getting a unique list of words
    """
    # lemma = WordNetLemmatizer()
    lemmatizer = nltk.WordNetLemmatizer().lemmatize
    text = re.sub('\W+', ' ', str(text))
    # print("step1:", text)
    text = re.sub(r'[0-9]+', '', text.lower())
    # correcting spellings of words - user complaints are bound to have spelling mistakes
    # text = TextBlob(text).correct()
    # print("step2:", text)
    word_pos = nltk.pos_tag(nltk.word_tokenize(text))
    normalized_text_lst = [lemmatizer(x[0], get_wordnet_pos(x[1])).lower() for x in word_pos]
    # print("step3:", normalized_text_lst)
    stop_words_free = [i for i in normalized_text_lst if i not in english_stopwords and len(i) > 3]
    # print("step4:", stop_words_free)
    stop_words_free = list(set(stop_words_free))
    return (stop_words_free)


def get_average_word2vec(complaints_lst, model, num_features=300):
    """
    Function to average the vectors in a list.
    Say a list contains 'flower' and 'leaf'. Then this function gives - model[flower] + model[leaf]/2
    - index2words gets the list of words in the model.
    - Gets the list of words that are contained in index2words (vectorized_lst) and
      the number of those words (nwords).
    - Gets the average using these two and numpy.
    """
    index2word_set = set(model.wv.index2word)
    vectorized_lst = []
    vectorized_lst = [model[word] if word in index2word_set else np.zeros(num_features) for word in \
                      complaints_lst]
    nwords = len(vectorized_lst)
    summed = np.sum(vectorized_lst, axis=0)
    averaged_vector = np.divide(summed, nwords)
    return averaged_vector


def process_text(complaint):
    print("Inside prediction")
    num_features = 300

    print("Path:", os.path.dirname(os.path.abspath(__file__)))

    if not complaint:
        return "We're happy you have no complaint! If you pressed Submit by mistake, go back."

    print("Cleaning...")
    cleaned_complaint = clean_up(complaint)
    if not cleaned_complaint:
        return "Sorry, your complaint contains words on which the model isn't trained. Time to call a human."

    print("Loading Word2Vec model...")
    #word2vec_model = KeyedVectors.load_word2vec_format('../trained_models/GoogleNews-vectors-negative300.bin', binary=True)
    word2vec_model = gensim.models.Word2Vec.load("trained_models/300features_10minwords_10context1")

    print("Getting embeddings...")
    embeddings = get_average_word2vec(cleaned_complaint, word2vec_model, num_features)
    input_df = pd.DataFrame(embeddings).T
    # if embeddings.size == 0:
    #     #return "Sorry, your complaint contains words on which the model isn't trained. Time to call a human."
    #     print("Sorry, your complaint contains words on which the model isn't trained. Time to call a human.")

    print("Predicting from a pre-trained model..")
    trained_model = pickle.load(open('../trained_models/rf_word2vec_model_6530.model', 'rb'))
    prediction = trained_model.predict(input_df)

    rf_pred_prob = trained_model.predict_proba(input_df)
    print("Result=", prediction[0], np.amax(rf_pred_prob))
    return prediction[0], round(np.amax(rf_pred_prob),2)