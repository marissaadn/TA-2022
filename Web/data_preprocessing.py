import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
# pip install PySastrawi
# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Content = "Video berisi liputan pemberitaan Kompas TV berjudul Awas Vaksin Palsu menyebar di grup Whatsapp dan Twitter sejak Rabu malam 21 Juli 2021. Berita ini beredar di tengah program vaksinasi Covid-19 di Indonesia.Dalam video berdurasi 2 menit 35 detik itu, diberitakan bahwa pasangan suami istri asal Bekasi, Jawa Barat ditangkap karena memproduksi vaksin palsu.DiTwitter, video tersebut beredar dengan narasi, â€œPasutri Pembuat VAKSIN PALSU diTANGKAP. Manusia tdk berguna yg mencelakakan org lain sangat LAYAK diHUKUM seberat-beratnya,kalo perlu hukuman seumur hidup.â€Sedangkan di WhatsApp, video disebarkan tanpa konteks dan keterangan apa-apa."
#Case Folding
def case_folding(text):
    text = text.lower()
    return text

# Content = case_folding(Content)

# Noise Removal
import re
def remove_noise(text):
    # hapus tab, new line, back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\',"")
    # hapus non ASCII (emoticon, chinese word, dll)
    text = text.encode('ascii', 'replace').decode('ascii')
    # hapus @ mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    #hapus angka
    text= re.sub('[0-9]+', '', text)
    return text

# Content = remove_noise(Content)

#remove punctuation
import string 
def remove_punctuation(text):
    transtable = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.translate(transtable)

# Content = remove_punctuation(Content)

# Hapus multiple space
def remove_multiple_space(text):
    return re.sub('\s+',' ',text)

# Content = remove_multiple_space(Content)

def word_tokenization(text):
    return word_tokenize(text)

# Content = word_tokenization(Content)

nltk.download('stopwords')
from nltk.corpus import stopwords

    # stopword indonesia
list_sw = stopwords.words('indonesian')

    # tambahan stopword
list_sw.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', "20detik", "baca"])

    # tambahan stopword
# txt_sw = pd.read_csv("stopwords-id.txt", names= ["stopwords"], header = None)

    # stopword string -> list & append tambahan stopword
# list_sw.extend(txt_sw["stopwords"][0].split(' '))

    # list -> dictionary
list_sw = set(list_sw)

# stopword removal
def stopword_removal(words):
    return [word for word in words if word not in list_sw]

# Content = stopword_removal(Content)


# normalisasi = pd.read_csv('new_colloquial-indonesian-lexicon.csv')

# normalisasi_kata_dict = {}

# for index, row in normalisasi.iterrows():
#     if row[0] not in normalisasi_kata_dict:
#         normalisasi_kata_dict[row[0]] = row[1] 

# def normalized_term(document):
#      return [normalisasi_kata_dict[term] if term in normalisasi_kata_dict else term for term in document]



# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

def termdict(Content):
    term_dict = {}

    # for document in df['Content']:
    for term in Content:
        if term not in term_dict:
            term_dict[term] = ' '
    
            
    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])
        
    print(term_dict)
    print("------------------------")
    return term_dict


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

# Content = get_stemmed_term(Content)

# print(Content)
# df['ulasan'] = df['ulasan'].swifter.apply(get_stemmed_term)
# df['ulasan'].head(10)

# df.to_csv("Data_Test_Preprocessed.csv", index=False)