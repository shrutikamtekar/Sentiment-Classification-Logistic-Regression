from sentiment_reader import SentimentCorpus
from logistic_regression import LogisticRegression

if __name__ == '__main__':
    dataset = SentimentCorpus()
    lr = LogisticRegression()

    params, predict_train = lr.train(dataset.train_X,dataset.train_y)
    eval_train, MacroAvgF1_train = lr.accuracy_fscore(predict_train,dataset.train_y)
    
#    predict_test = lr.test(dataset.test_X,dataset.test_y,params)
    predict_test = lr.test(dataset.test_X,params)
    eval_test, MacroAvgF1_test = lr.accuracy_fscore(predict_test,dataset.test_y)
    
   
    print "Accuracy on training set: %f, on test set: %f" % (eval_train, eval_test)
    print ""
    print "Macro Averaged F1 score on training set: %f, on test set: %f" % (MacroAvgF1_train, MacroAvgF1_test)


