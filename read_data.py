import codecs

def preprocess(s):
    s=s.lower()
    s_new=""
    for i in s:
        if i in "abcdefghijklmnopqrstuvwxyz \n":
            s_new+=i
        else:
            s+=" "
    return s_new

def remove_stopwords(d):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.add("subject")
    new_d =[]
    for i in d:
        if i not in stop_words:
            new_d.append(i)
    return new_d

def counts(d):
    d = remove_stopwords(d)
    count={}
    for i in d:
        if i in count.keys():
            count[i]+=1
        else:
            count[i]=1
    return count

def laplace_smoothening(w,d,n):
    for i in w:
        if i not in d:
            d[i]=n
        else:
            d[i]+=n
    return d

def fnames(mypath):
    from os import walk
    f=[]
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        return f

def read_data(mypath):
    
    words = []
    word_count = {}
    # mypath="./train/spam/"
    f = fnames(mypath)
    data=""
    prior=len(f)
    # import codecs
    for i in f:
        fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                          errors='ignore')
        data+=preprocess(fin.read())
        data+=" "
    words = data.split()
    word_count=counts(data.split())
    # print(spam_words)
    return prior,words,word_count

    # print(spam)

def get_allwords():
    allwords = []
    paths = ["./test/ham/", "./test/spam/", "./train/ham/", "./train/spam/"]
    for mypath in paths:
        f = fnames(mypath)
        for i in f:
            fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                            errors='ignore')
            sen = remove_stopwords(preprocess(fin.read()).split())
            allwords.extend(sen)
    return set(allwords)

# print(counts("i am a beg beg student i am not in a nug me please tea".split()))
