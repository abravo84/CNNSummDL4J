This package contains the source code and sample data related to the Coling 2018 submission entitle: "Unlocking sentence relevance with convolutions for citation-based text summarization"

The source code is available in the CnnSentencePredictionFinal.java file. The parameters are (in this order):

	args[0]: Path of the file containing w2v information, such as googleacl_vec_sw_15.txt (for single channel) or googleacl_vec_sw_15.txt (dual channel)
 or     args[1]: Path of the file containing context features, such as add_feat_vec_max_window.csv
    	args[2]: Path of the scoring function (such as )
    	args[3]: Height of the Matrix from w2v (15);
    	args[4]: Width of the Matrix from w2v (300);
    	args[5]: number of iterations in the CNN in the training.
    	args[6]: learning_rate
    	args[7]: number of ecpochs
        args[8]: bach size
	args[9]: number of filters
	args[10]: name of the experiment (it will append to the result path)
        args[11]: we defiend diferent types of networks configurations;
        args[12]: Number to classes to learn (1 for regression task)
        args[13]: number of inputs to be considered (1, 2 or 2 in dual channel)
        args[14]: window filter size of the first convolution (it is 2);
	
	args[15]: Boolean indicating if you wast training and testing (True) or only training (False)
        
        args[16]: Path of the Test dataset.
        args[17]: Prefix of the test file containing w2v information such as "googleacl_vec_sw_15_" (for single channel) or "googleacl_vec_sw_15_" (dual channel)
        args[18]: Prefix of the test file containing context features, such as "add_feat_vec_max_window_".
        args[19]: Output folder for results;
        

The sample data is located in sample_dataset where:

	train_100_rnd_instances folder: contains files for training.
	test_2_docs folder: contains files for training.


