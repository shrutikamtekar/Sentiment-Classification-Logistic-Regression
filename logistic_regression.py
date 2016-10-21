import numpy as np
import theano
import theano.tensor as T

class LogisticRegression():
    def train(self, train_x, train_y):
        #getting the no of instances in x and no of features in x
        n_instances, n_feats= train_x.shape
        #no of classes in y
        n_classes = np.unique(train_y).shape[0]
        #no of times to iterate
        n_epoches = 1500
        
        #convert 2d into 1d because y is a vector
        train_y = train_y.ravel()
        
       
        # declare Theano symbolic variables
        x = T.matrix("x")
        y = T.ivector("y")
#        w = theano.shared(np.random.randn(n_feats,n_classes), name="w")
        w = theano.shared(np.zeros((n_feats,n_classes),dtype=theano.config.floatX),name='W')
        b = theano.shared(np.zeros(n_classes), name="b")

        print("Initial model for w:")
        print(w.get_value())
        print("Initial model for b:")
        print(b.get_value())

        # construct Theano expression graph
        p_y_given_x = T.nnet.softmax(T.dot(x, w) + b)
        xent = -T.mean(T.log(p_y_given_x)[T.arange(n_instances), y])
        cost = xent + 0.01 * (w ** 2).sum()       # The cost to minimize
        gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
        y_pred = T.argmax(p_y_given_x, axis=1)
        error = T.mean(T.neq(y_pred, y))
         
        # compile
        train = theano.function(inputs=[x,y],
          outputs=[error, cost, y_pred],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
        
        #computing error and cost
        print ""
        print "Computing error and cost"
        # train
        for i in range(n_epoches):
            error, cost, y_pred = train(train_x, train_y)
#            print 'Current error: %.4f | Current cost: %.4f' % (error, cost)

        print 'Final error: %.4f | Final cost: %.4f' % (error, cost)    
        print ""
        print("Final model for w:")
        print(w.get_value())
        print("Final model for b:")
        print(b.get_value())
        print ""
        
        return [w,b],y_pred
   
    def test(self,test_x,params):
        #getting the no of instances in x and no of features in x
        n_instances, n_feats= test_x.shape
        
        w = params[0]
        b = params[1]
        
        x = T.matrix("x")
        
        # construct Theano expression graph
        p_y_given_x = T.nnet.softmax(T.dot(x, w) + b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        
        test = theano.function(inputs=[x],
           outputs=[y_pred])
         
        y_pred = test(test_x)
        y_pred = np.asarray(y_pred).transpose() #to make the size similar to test_y to calculate accuracy and F1Score
        
        return y_pred
        
    def accuracy_fscore(self,y_pred,data_y):
        correct = 0.0
        total = 0.0
        n_classes=np.unique(data_y).shape[0]
        actual_predicted = np.ndarray(shape=(n_classes,n_classes))
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        fscore = np.zeros(n_classes)
        
        for i in range(len(data_y)):
            if(data_y[i] == y_pred[i]):
                correct += 1
            if (data_y[i] ==0 and y_pred[i] == 0):
                 actual_predicted[0,0]+=1
            elif (data_y[i] ==0 and y_pred[i] == 1):
                 actual_predicted[0,1]+=1
            elif (data_y[i] ==1 and y_pred[i] == 0):
                 actual_predicted[1,0]+=1
            elif (data_y[i] ==1 and y_pred[i] == 1):
                 actual_predicted[1,1]+=1
            total += 1
        
        precision[0] = actual_predicted[0,0]/(actual_predicted[0,0]+actual_predicted[1,0])
        precision[1] = actual_predicted[1,1]/(actual_predicted[0,1]+actual_predicted[1,1])
        
        recall[0]= actual_predicted[0,0]/(actual_predicted[0,0]+actual_predicted[0,1])
        recall[1]= actual_predicted[1,1]/(actual_predicted[1,0]+actual_predicted[1,1])
        
        fscore[0] = (2*precision[0]*recall[0])/(precision[0]+recall[0])
        fscore[1] = (2*precision[1]*recall[1])/(precision[1]+recall[1])
        
            
        return 1.0*correct/total, np.average(fscore)


	
	
