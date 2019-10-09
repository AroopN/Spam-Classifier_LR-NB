from read_data import *

if __name__ == '__main__':
    spam_words, spam_word_counts = read_data("./train/spam/")
    ham_words, ham_word_counts = read_data("./train/ham/")

    all_words = get_allwords()
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


    mypath = "./test/spam/"
    f = fnames(mypath)
    X = []
    y = []
    for i in f:
        fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                        errors='ignore')
        X.append(remove_stopwords(preprocess(fin.read()).split()))
        y.append(1)


    mypath = "./test/ham/"
    f = fnames(mypath)
    X = []
    y = []
    for i in f:
        fin = codecs.open(mypath+i, 'r', encoding='utf-8',
                        errors='ignore')
        sen = remove_stopwords(preprocess(fin.read()).split())
        X.append(sen)
        y.append(0)

    # print(X[10])
    pred = []
    for i in X:
        sp = 0.5
        hp = 0.5
        for j in i:
            try:
                sp *= spam_word_prob[j]
                hp *= ham_word_prob[j]
            except:
                print(j)
        if sp > hp:
            pred.append(1)
        else:
            pred.append(0)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(pred, y))

    # print(sorted(spam_word_counts.items(), key=lambda kv: kv[1]))


