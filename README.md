# Sentiment-Classification-Logistic-Regression  
Implementation of Logisitc Regression classifier using theano toolkit and apply it to sentiment classification task.  

### Description  
<p>There is positive.review and negative.review are two data files that respectively contain
positive and negative book reviews. Each line in the file corresponds to a review document. Each token
(e.g., year:2) in the line corresponds to a word and its frequency in the document. The last token (e.g.,
#label#:negative) in each line indicates the polarity (label) of the document.</p> 

**On executing the run_classifier.py file, it will return the following results:**   
Initial model for w:  
[[ 0.  0.]  
 [ 0.  0.]  
 [ 0.  0.]  
 ...,   
 [ 0.  0.]  
 [ 0.  0.]  
 [ 0.  0.]]  
Initial model for b:  
[ 0.  0.]  

Computing error and cost  
Final error: 0.0050 | Final cost: 0.2762  

Final model for w:  
[[-0.01170453  0.01170453]  
 [ 0.00264991 -0.00264991]  
 [-0.00364406  0.00364406]  
 ...,   
 [-0.00408958  0.00408958]  
 [ 0.02015603 -0.02015603]  
 [ 0.00185579 -0.00185579]]  
Final model for b:  
[ 0.01997021 -0.01997021]  

Accuracy on training set: 0.995000, on test set: 0.842500  
Macro Averaged F1 score on training set: 0.969790, on test set: 0.964699  
