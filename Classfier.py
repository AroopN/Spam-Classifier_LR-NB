from read_data import *
from math import log

if __name__ == '__main__':
    rem_stopwords=0
    spam_prior, ham_prior, spam_word_prob, ham_word_prob = trainNB(
        rem_stopwords)
    mypath = "./test/spam/"
    f = fnames(mypath)
    X = []
    y = []
    for i in f:
        fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                        errors='ignore')
        sen = preprocess(fin.read()).split()
        if rem_stopwords == 1:
            sen = remove_stopwords(sen)
        X.append(sen)
        y.append(1)


    mypath = "./test/ham/"
    f = fnames(mypath)
    X = []
    y = []
    for i in f:
        fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                        errors='ignore')

        sen = preprocess(fin.read()).split()
        if rem_stopwords==1:
            sen=remove_stopwords(sen)
        X.append(sen)
        y.append(0)

    # print(X[10])
    # print(spam_prior)
    pred = []
    for i in X:
        sp = log(spam_prior)
        hp = log(ham_prior)
        for j in i:
            try:
                sp += log(spam_word_prob[j])
                hp += log(ham_word_prob[j])
            except:
                print(j)
        if sp > hp:
            pred.append(1)
        else:
            pred.append(0)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred, y))

    # print(sorted(spam_word_counts.items(), key=lambda kv: kv[1]))


