import numpy as np
from gensim import corpora, models
from sklearn.svm import SVC
from data_process import ReadData, Dataset

def LDA(train_data, train_label, test_data, test_label,num_topics = 100):
    dictionary = corpora.Dictionary(train_data)
    lda_corpus_train = [dictionary.doc2bow(tmp_doc) for tmp_doc in train_data]
    lda = models.LdaModel(corpus=lda_corpus_train, id2word=dictionary, num_topics = num_topics)

    #### train svm classifier for correct label
    train_topic_distribution = lda.get_document_topics(lda_corpus_train)
    train_features = np.zeros((len(train_data), num_topics))
    for i in range(len(train_topic_distribution)):
        tmp_topic_distribution = train_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            train_features[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]

    assert len(train_label) == len(train_features)
    train_label = np.array(train_label)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(train_features, train_label)
    print("训练集准确率： {:.4f}.".format(sum(classifier.predict(train_features) == train_label) / len(train_label)))

    lda_corpus_test = [dictionary.doc2bow(tmp_doc) for tmp_doc in test_data]
    test_topic_distribution = lda.get_document_topics(lda_corpus_test)
    test_features = np.zeros((len(test_data), num_topics))
    for i in range(len(test_topic_distribution)):
        tmp_topic_distribution = test_topic_distribution[i]
        for j in range(len(tmp_topic_distribution)):
            test_features[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]
    assert len(test_label) == len(test_features)
    test_label = np.array(test_label)
    print("测试集准确率： {:.4f}.".format(sum(classifier.predict(test_features) == test_label) / len(test_label)))


if __name__ == '__main__':
    context = ReadData('dataset')
    traindata, trainlabel, testdata, testlabel = Dataset(context)
    LDA(traindata, trainlabel, testdata, testlabel)