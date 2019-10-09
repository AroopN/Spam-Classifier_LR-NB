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

def counts(d,stopwords=1):
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
    # word_count = {}
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
    # word_count=counts(data.split())
    # print(spam_words)
    return prior,words

    # print(spam)

def get_allwords():
    allwords = []
    paths = ["./test/ham/", "./test/spam/", "./train/ham/", "./train/spam/"]
    for mypath in paths:
        f = fnames(mypath)
        for i in f:
            fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                            errors='ignore')
            sen = preprocess(fin.read()).split()
            allwords.extend(sen)
    return set(allwords)

def trainNB(rem_stopwords=1):
    
    spams, spam_words = read_data("./train/spam/")
    spam_word_counts = counts(spam_words, rem_stopwords)
    hams, ham_words = read_data("./train/ham/")
    ham_word_counts = counts(ham_words, rem_stopwords)
    spam_prior = spams/(spams+hams)
    ham_prior = hams/(spams+hams)
    all_words = get_allwords()
    if rem_stopwords==1:
        all_words = remove_stopwords(all_words)

    spam_word_counts = laplace_smoothening(all_words, spam_word_counts, 1)
    ham_word_counts = laplace_smoothening(all_words, ham_word_counts, 1)
    # print(ham_words[99])

    tot_spam_words = sum(spam_word_counts.values())
    spam_word_prob = spam_word_counts
    for i in all_words:
        spam_word_prob[i] = spam_word_prob[i]/tot_spam_words

    tot_ham_words = sum(ham_word_counts.values())
    ham_word_prob = ham_word_counts
    for i in all_words:
        ham_word_prob[i] = ham_word_prob[i]/tot_ham_words

    return spam_prior, ham_prior, spam_word_prob, ham_word_prob
