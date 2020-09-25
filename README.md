# DLP-Lab4
## Sequence-to-sequence Recurrent Network
(The report please refer to DLP_LAB4_Report_0886035.pdf) 

#### Lab Objective
* In this lab, you need to implement a seq2seq recurrent neural network for English spelling correction.
* Spelling correction
* E.g. ‘recetion’ to ‘recession’


#### Lab Description
* To understand sequence-to-sequence architecture
* Embedding function
* Teacher forcing technique
* Word dropout


#### Architecture
* Input word: ‘recetion’
* Output word : ‘recession’

![Architecture](/picture/architecture.png "Architecture")

#### seq2seq
* Embedding function (for characters)
  * [1,2,3]  -> [shape(128), shape(128), shape(128)] (high dimensional space)
* Teacher forcing
  * Using ground truth character instead of the output of the decoder. 
* Word dropout (optional)
  * Change the input of the decoder to <UNK> token. 


#### Other details
* The encoder and decoder must be implemented by LSTM. 
* The loss function is nn.CrossEntropyLoss().
* The optimizer is SGD
* Adopt BLEU-4 score function in NLTK.
* Average all testing scores
* Save your training weights. 
* While demo, you should load your model weights and run the evaluation to get the results.



#### Requirements
* Modify encoder, decoder, and training functions
* Implement evaluation function and data 
* Plot the crossentropy loss and BLEU-4 score curve during training.
* Output examples:

![Requirements](/picture/Requirements.png "Requirements")
