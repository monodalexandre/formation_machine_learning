from flask import Flask, render_template, request

import pandas as pd
import gensim
import spacy
nlp = spacy.load("en_core_web_sm")
from joblib import load


from gensim.models import TfidfModel
from bs4 import BeautifulSoup

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Chargement du modèle nécessaire à l'API
best_model = load('best_model.pkl')
multilabel_binarizer_cv = load('multilabel_binarizer_cv.pkl')
cvect_vocabulary = load('cvect_vocabulary.pkl')

app = Flask(__name__)

@app.route('/requete',methods=['GET'])
def dashboard():

    # Affichage du template de saisie des données title et post
    return render_template('dashboard.html')

@app.route('/resultat',methods=['POST'])
def resultat():

    # Récupération des variables du formulaire
    result=request.form
    post_in = result['Post_content']
    print(post_in)
    

    # Transformation des données saisies en listes de mots (mêmes transformations que notebook d'anal. explo.)
    post_out = tokenization_full(post_in)
    #post_out = [post_out]

    
    # Model
    cvect = CountVectorizer(stop_words='english', max_df=0.95, min_df=10,
                vocabulary=cvect_vocabulary)
    cv_transform = cvect.fit_transform(post_out)
    ft_cv = cvect.get_feature_names_out()
    cv_data_post_in = pd.DataFrame.sparse.from_spmatrix(cv_transform, columns=ft_cv)
    cv_data_post_in.drop(columns="tag", inplace=True)

    #Prediction
    prediction_cv_format = best_model.predict(cv_data_post_in)
    prediction_tag  = multilabel_binarizer_cv.inverse_transform(prediction_cv_format)
    prediction_tag = [i for i in prediction_tag if i != ()]

    # Affichage du template de résultat
    return render_template('resultat.html', Post=post_in,
                           Tags=prediction_tag[0:5])


def suppr_balises_html(text):
    soup = BeautifulSoup(text, "html.parser")
    text_out = ' '.join(soup.stripped_strings)
    # Remove \n
    text_out = text_out.replace("\n", " ")        
    return text_out

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def replace_pos(tokens):
    k = 0
    for i in tokens.keys():
        tokens[i] = get_wordnet_pos(tokens[i])
        k += 1
    return tokens

def lemmatize(tokens):
    WNlemmatizer = WordNetLemmatizer()
    lem_tokens = []
    for key in tokens.keys():
        if tokens[key] is None :    # In case there are no tags
            lem_tokens.append(key)
        else :
            lem_tokens.append(WNlemmatizer.lemmatize(key, pos=tokens[key]) )
    return lem_tokens

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele + ' '
    return str1

def keep_nouns(sentence):
    liste = []
    for word in sentence:
        if word.pos_ == "NOUN" or word.pos_ == "PROPN":
            liste.append(word)
    return liste

def tokenization_full(text):
    text_out = suppr_balises_html(text)
    text_out = text_out.lower()
    # Tokenize
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text_out)
    # Stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    text_without_sw = [word for word in tokenized if word not in stopwords and len(word) > 2]
    # First need to POS-tag tokens
    postagged = dict(nltk.pos_tag(text_without_sw))
    tokens = replace_pos(postagged)
    lem_tokens = lemmatize(tokens)
    # NER
    tokens_from_list_to_strings = listToString(lem_tokens)
    sentences_nlp_ = nlp(tokens_from_list_to_strings)
    ner_tokens = keep_nouns(sentences_nlp_)
    detokenized = [token.text_with_ws for token in ner_tokens]
    return detokenized


if __name__ == "__main__":
        app.run()
