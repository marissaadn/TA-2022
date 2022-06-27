from flask import Flask, render_template, request
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string 
nltk.download('punkt')
from knn import *

app = Flask(__name__)

target = {0:'Fakta', 1:'Hoax'}

def case_folding(text):
    text = text.lower()
    return text

def remove_noise(text):
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    text= re.sub('[0-9]+', '', text)
    return text

def remove_punctuation(text):
    transtable = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.translate(transtable)

def remove_multiple_space(text):
    return re.sub('\s+',' ',text)

def word_tokenization(text):
    return word_tokenize(text)

nltk.download('stopwords')
from nltk.corpus import stopwords
list_sw = stopwords.words('indonesian')
list_sw.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', "20detik", "baca"])
txt_sw = pd.read_csv("stopwords-id.txt", names= ["stopwords"], header = None)
list_sw.extend(txt_sw["stopwords"][0].split(' '))
list_sw = set(list_sw)

def stopword_removal(words):
    return [word for word in words if word not in list_sw]

normalisasi = pd.read_csv('new_colloquial-indonesian-lexicon.csv')
normalisasi_kata_dict = {}
for index, row in normalisasi.iterrows():
    if row[0] not in normalisasi_kata_dict:
        normalisasi_kata_dict[row[0]] = row[1] 

def normalized_term(document):
     return [normalisasi_kata_dict[term] if term in normalisasi_kata_dict else term for term in document]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

@app.route('/', methods=["GET", "POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        df = pd.DataFrame([inp], columns=['content'])
        df['content'] = df['content'].apply(case_folding)
        df['content'] = df['content'].apply(remove_noise)
        df['content'] = df['content'].apply(remove_punctuation)
        df['content'] = df['content'].apply(remove_multiple_space)
        df['content'] = df['content'].apply(word_tokenization)
        df['content'] = df['content'].apply(stopword_removal)
        df['content'] = df['content'].apply(normalized_term)
        # inp = case_folding(inp)
        # inp = remove_noise(inp)
        # inp = remove_punctuation(inp)
        # inp = remove_multiple_space(inp)
        # inp = word_tokenization(inp)
        # inp = stopword_removal(inp)
        # inp = normalized_term(inp)

        term_dict = {}
        for document in df['content']:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = ' '
        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)
        def get_stemmed_term(document):
            return [term_dict[term] for term in document]

        df['content'] = df['content'].apply(get_stemmed_term)

        tfvectorizer = joblib.load('models/tfidf.pkl')
        inp_tf = tfvectorizer.transform(" ".join(row) for row in df['content'].values)

        data_test = pd.DataFrame(
            inp_tf.todense(), 
            columns=tfvectorizer.get_feature_names_out()
        )
        unselected_feature_list = pd.read_csv("unselected_feature_list.csv")
        df_test_with_feature_selection = data_test.drop(columns=unselected_feature_list["0"].values)
        X_test = df_test_with_feature_selection.values

        knn_pred = joblib.load('models/my_model_knn.pkl')

        pred = knn_pred.predict(X_test, 3, 'euclidean')

        if pred[0]==0:
            desc = 'FAKTA'
        elif pred[0]==1:
            desc = 'HOAX'
        return render_template('index.html', message = desc, text_news = inp)
        # if pred == 1:
        #     return render_template('index.html', message = "Hoax!!!")
        # else:
        #     return render_template('index.html', message = "Fakta!!!")

    return render_template('index.html')

# app = Flask(__name__)
# model = tf.keras.models.load_model('model.h5')
# model.make_predict_function()

# @app.route('/', methods=['GET'])
# def main():
#     return render_template('index.html')
    
# @app.route('/', methods=['POST'])
# def predict():
#     imagefile = request.files['imagefile']
#     image_path = "./static/" + imagefile.filename
#     imagefile.save(image_path)

#     image = load_img(image_path, target_size=(150,150))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     pred = model.predict(image)
#     if pred[0][0]>0:
#         desc = 'NORMAL'
#     elif pred[0][1]>0:
#         desc = 'TUBERCULOSIS'

#     classification = '%s' % (desc)

#     return render_template('index.html', prediction=classification, image=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)