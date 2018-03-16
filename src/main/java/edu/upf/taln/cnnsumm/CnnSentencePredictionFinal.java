package edu.upf.taln.cnnsumm;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.*;
import java.util.*;

/**
 * Convolutional Neural Networks for Sentence Classification - https://arxiv.org/abs/1408.5882
 *
 * Specifically, this is the 'static' model from there
 *
 * @author Alex Black
 */
public class CnnSentencePredictionFinal {

    public static final int nChannels = 1;
    public static int height = 75;
    public static int width = 303;
    public static int classes = 4;
    public static final int seed = 123;
    public static int iterations = 1;
    public static int nEpochs = 1;
    public static int miniBatchSize = 256;
    public static double learningRate = 1e-3;
    public static String algorithType="";
    public static String inputMode="";
    public static int window = 256;
    public static int nFilters = 256;
    public static boolean testMode = false;
    public static int nClusters = 0;
    
    public static MultiNormalizerMinMaxScaler minMaxScaler;
    
    public static void main(String[] args) throws Exception {

    	String trainingDataPath = "";
    	trainingDataPath = args[0];
    	String featuresAddPath = "";
    	featuresAddPath = args[1];
    	String labelsPath = "";
    	labelsPath = args[2];
    																																																																																																																																
    	height = Integer.parseInt(args[3]);
    	width = Integer.parseInt(args[4]);
    	iterations = Integer.parseInt(args[5]);
    	learningRate = Double.parseDouble(args[6]);
    	nEpochs = Integer.parseInt(args[7]);
        miniBatchSize = Integer.parseInt(args[8]);
        algorithType = args[11];
        classes = Integer.parseInt(args[12]);
        inputMode = args[13];
        window = Integer.parseInt(args[14]);
        nFilters = Integer.parseInt(args[9]);
        
        String testModeStr = args[15];
        
        String testDataPath = "";
        String vecPrefix = "";
        String featPrefix = "";
        String resultPath ="";
        if (testModeStr.toLowerCase().contains("t")) {
        	testMode = true;
        	testDataPath = args[16];
        	vecPrefix = args[17];
        	featPrefix = args[18];
        	resultPath = args[19];
        }
        //String citationsPath = args[20];
        //nClusters = Integer.parseInt(args[21]);
        //String citationPrefix = args[22];
        System.out.println("-------------------------------------");
        System.out.println("ARGUMENTS");
        System.out.println("-------------------------------------");
        System.out.println("height: " + height);
        System.out.println("width: " + width);
        System.out.println("iterations: " + iterations);
        System.out.println("learningRate: " + learningRate);
        System.out.println("nEpochs: " + nEpochs);
        System.out.println("miniBatchSize: " + miniBatchSize);
        System.out.println("algorithType: " + algorithType);
        System.out.println("classes: " + classes);
        System.out.println("inputMode: " + inputMode);
        System.out.println("window: " + window);
        System.out.println("nFilters: " + nFilters);
        System.out.println("-------------------------------------");
        
        
        //String featuresType = args[9]; // token_vec, token_google_vec, acl_vec
        
        String experimentNumber = args[10];//"_01";
        
        String reg = "reg";
        if (classes>1)
        	reg="class";
        
        experimentNumber = experimentNumber + "_a" + algorithType +  "_i" +inputMode +  "_w" +window +  "_e" +nEpochs +  "_lr" + learningRate +  "_" + reg;
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        
        //String trainingDataPath = "/home/upf/corpora/SciSUM-2017-arffs-and-models/training_sets_ab/ALL-DOCS-2016-TRAIN-27-FILES/token_acl_vec_human.txt";
        
        if (testMode){
        	ComputationGraph model = doExperimentTesting(trainingDataPath, featuresAddPath, labelsPath);
        	predictionCG(model, testDataPath, vecPrefix, featPrefix, resultPath, experimentNumber);
        }
        else {
        	doExperimentTraining(trainingDataPath, featuresAddPath, labelsPath);
        }
    }
    
    public static void predictionCG(ComputationGraph model, String testDataPath, String vecPrefix, String featPrefix, String testPathFinal, String experimentNumber) throws IOException, InterruptedException {
    	int totalRow = height*width;
    	File inDir=new File(testDataPath);
        File[] flist=inDir.listFiles();
        
        Arrays.sort(flist);

        //String testPathFinal = testDataPath + experimentNumber;
        
        File directory = new File(testPathFinal);
        if (! directory.exists()){
            directory.mkdir();
        }
        
        testPathFinal = testPathFinal + "/"+ experimentNumber;
        directory = new File(testPathFinal);
        if (! directory.exists()){
            directory.mkdir();
        }
        
        HashSet<String> docidDone = new HashSet<String>();
        
        for (File f: flist) {
        	String testFilePath = f.getAbsolutePath();
        	
        	String[] strList = testFilePath.split("_");
            String docID = strList[strList.length-1].replaceAll(".csv", "");
            
            if (docidDone.contains(docID))
            	continue;
        	
            docidDone.add(docID);
            
            String w2vPath = testDataPath + File.separator + vecPrefix + docID;    
            String featuresAddPath = testDataPath + File.separator + featPrefix + docID + ".csv";
            //String citationsPath = testDataPath + File.separator + citationPrefix + docID + ".csv";
            
            
            System.out.println("w2vPath FILE: " + w2vPath + "   " + countLines(w2vPath));
            System.out.println("featuresAddPath FILE: " + featuresAddPath + "   " + countLines(featuresAddPath));
            //System.out.println("citationsPath FILE: " + citationsPath + "   " + countLines(citationsPath));
            		
            
            int testBatch = countLines(w2vPath);
        	RecordReader word2vecRecord = new CSVRecordReader(0, ',');
        	word2vecRecord.initialize(new FileSplit(new File(w2vPath)));
        	
        	MultiDataSetIterator testIter = null;
        	
	    	if (inputMode.equals("1")) {
	    		testIter = new RecordReaderMultiDataSetIterator.Builder(testBatch)
	    		.addReader("word2vec", word2vecRecord)
	    		//.addReader("addFeat", addFeatRecord)
	    		.addInput("word2vec", 1, totalRow)
	    		//.addInput("addFeat",1,totalRow)
	    		.build();
	    	
	    	}else if (inputMode.equals("2")){
	    		RecordReader addFeatRecord = new CSVRecordReader(0, ',');
	        	addFeatRecord.initialize(new FileSplit(new File(featuresAddPath)));
	    		testIter = new RecordReaderMultiDataSetIterator.Builder(testBatch)
	    		.addReader("word2vec", word2vecRecord)
	    		.addReader("addFeat", addFeatRecord)
	    		.addInput("word2vec", 1, totalRow)
	    		.addInput("addFeat",1,30*7)
	    		.build();
	    	}else if (inputMode.equals("2d")) {
	    		RecordReader addFeatRecord = new CSVRecordReader(0, ',');
	        	addFeatRecord.initialize(new FileSplit(new File(featuresAddPath)));
	    		testIter = new RecordReaderMultiDataSetIterator.Builder(testBatch)
	    		.addReader("word2vec", word2vecRecord)
	    		.addReader("addFeat", addFeatRecord)
	    		.addInput("word2vec", 1, totalRow*2)
	    		.addInput("addFeat",1,30*7)
	    		.build();
	        	}
			
	    		
	    	
	    	if (classes==1) {
	    		testIter.setPreProcessor(minMaxScaler);
	    	}
    	    	
            
            MultiDataSet testData = testIter.next();
            
            //System.out.println(model.output(false, testData.getFeatures()));
            INDArray[] predictedM = model.output(false, testData.getFeatures());
            //if (classes == 1) {
            // 	predicted = model.output(false, testData.getFeatures());
            //}else {
            
            //INDArray predicted = model.outputSingle(false, testData.getFeatures());
            INDArray predicted = model.output(false, testData.getFeatures())[0];
            
            if (classes==1) {
            	minMaxScaler.revertLabels(predictedM);
            	predicted = predictedM[0];
	    	}
            
            //}
            //normalizer.revertLabels(predicted);
            String resultPath  = "";
            //resultPath = testFilePath.replace(testDataPath, testPathFinal)+ "_" + optionFeature+".txt";
            //resultPath = testFilePath.replace(testDataPath, testPathFinal).replace(featuresType, featuresType + "_" + optionFeature).replace(".txt", ".csv.txt");
            
            resultPath = testPathFinal + File.separator + docID;
            if (!resultPath.endsWith(".csv"))
            	resultPath = resultPath + ".csv";
            
            PrintWriter pw=new PrintWriter(new FileWriter(resultPath));

            System.out.println("Result File:  " + resultPath);
            //System.out.println("predicted:  " + predicted);
            
            if (classes == 1) {
            	for (int i = 0; i < predicted.length(); i++) {
            		pw.println(i + "\t" + predicted.getFloat(i));
            		pw.flush();
            	}
            }else {
            	pw.println(predicted);
            	pw.flush();
            	pw.close();
            }
        }
    	
    }

public static MultiDataSetIterator getTrainData(String trainingDataPath, String featuresAddPath, String labelsPath) throws IOException, InterruptedException {
	//trainingDataPath = featuresPath + File.separator + "acl_vec_"+height+".txt";
	int totalRow = height*width;
	RecordReader word2vecRecord = new CSVRecordReader(0, ',');
	word2vecRecord.initialize(new FileSplit(new File(trainingDataPath)));
	
	RecordReader addFeatRecord = new CSVRecordReader(0, ',');
	addFeatRecord.initialize(new FileSplit(new File(featuresAddPath)));
	
	
	RecordReader labelsRecord = new CSVRecordReader(0, ',');
	labelsRecord.initialize(new FileSplit(new File(labelsPath)));
	MultiDataSetIterator trainIter = null;
	
	if (classes==1) {
    	if (inputMode.equals("1")) {
    	
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
	    		.addReader("word2vec", word2vecRecord)
	    		.addReader("labels", labelsRecord)
	    		.addInput("word2vec", 1, totalRow)
	    		.addOutput("labels", 1, 1) 
	    		.build();
    	
    	}else if (inputMode.equals("2")) {
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
	    		.addReader("word2vec", word2vecRecord)
	    		.addReader("addFeat", addFeatRecord)
	    		.addReader("labels", labelsRecord)
	    		.addInput("word2vec", 1, totalRow)
	    		.addInput("addFeat",1,30*7)
	    		.addOutput("labels", 1, 1) 
	    		.build();
    	}else if (inputMode.equals("2d")) {
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
    	    		.addReader("word2vec", word2vecRecord)
    	    		.addReader("addFeat", addFeatRecord)
    	    		.addReader("labels", labelsRecord)
    	    		.addInput("word2vec", 1, totalRow*2)
    	    		.addInput("addFeat",1,30*7)
    	    		.addOutput("labels", 1, 1) 
    	    		.build();
        	}
		
    	
	}else {
		if (inputMode.equals("1")) {
	    	
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
	    		.addReader("word2vec", word2vecRecord)
	    		//.addReader("addFeat", addFeatRecord)
	    		.addReader("labels", labelsRecord)
	    		.addInput("word2vec", 1, totalRow)
	    		//.addInput("addFeat",1,totalRow)
	    		.addOutputOneHot("labels", 1, classes)
	    		.build();
	    	
    	}else if (inputMode.equals("2")){
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
	    		.addReader("word2vec", word2vecRecord)
	    		.addReader("addFeat", addFeatRecord)
	    		.addReader("labels", labelsRecord)
	    		.addInput("word2vec", 1, totalRow)
	    		.addInput("addFeat",1,30*7)
	    		.addOutputOneHot("labels", 1, classes)
	    		.build();
    	}else if (inputMode.equals("2d")){
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
    	    		.addReader("word2vec", word2vecRecord)
    	    		.addReader("addFeat", addFeatRecord)
    	    		.addReader("labels", labelsRecord)
    	    		.addInput("word2vec", 1, totalRow*2)
    	    		.addInput("addFeat",1,30*7)
    	    		.addOutputOneHot("labels", 1, classes)
    	    		.build();
        	}
		
	}
	
	
	if (classes==1) {
    	minMaxScaler = new MultiNormalizerMinMaxScaler();
    	minMaxScaler.fitLabel(true);
    	minMaxScaler.fit(trainIter);
    	trainIter.reset();
    	trainIter.setPreProcessor(minMaxScaler);
	}
	
	
	return trainIter;
	
}


public static INDArray getWeights(Word2Vec w2v, int vocabSize, int dimension) { 
	
	INDArray rows = Nd4j.createUninitialized(new int[]{vocabSize + 1, dimension}, 'c');
	
	INDArray padding = Nd4j.zeros(dimension);
	
	rows.putRow(0, padding);
	for (int i = 0; i < vocabSize; i++) {
		String word = w2v.getVocab().wordAtIndex(i);
	    double[] embeddings = w2v.getWordVector(word); // getEmbeddings is my own function
	    INDArray newArray = Nd4j.create(embeddings);
	    rows.putRow(i+1, newArray);
	}
	return rows;
	
}

public static ComputationGraphConfiguration getNNConfig(int vocasize) {
	return getNetworkConfigurationBoth2Regr4CNN_Embedding(vocasize);
}

public static ComputationGraphConfiguration getNNConfig() {
	ComputationGraphConfiguration conf = null;
    
    
    switch (algorithType) {
    case "1i_1CNN":
    	System.out.println("--> oldCNN");
    	if (classes==1)
    		conf = getNetworkConfigurationWindowRegre();
    	else
    		conf = getNetworkConfigurationWindowClass();
    	break;
    	
    case "2i_1CNN":
    	System.out.println("--> newCNN");
    	if (classes==1)
    		conf = getNetworkConfigurationAlLoro2Regre();
    	else
    		conf = getNetworkConfigurationAlLoro2Class();
    	break;
    	
    case "2i_2CNN":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2Regr();
    	else
    		conf = getNetworkConfigurationBoth2();
    	break;
    case "2i_2CNNsimple":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2Regr();
    	else
    		conf = getNetworkConfigurationBoth2Simple();
    	break;	
    case "2i_2CNNplus":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2Regr4CNN();
    	else
    		conf = getNetworkConfigurationBoth2_4CNN();
    	break;	
    case "2i_2CNNdual":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrDual();
    	else
    		conf = getNetworkConfigurationBoth2Dual();
    	break;
    	
    case "2i_2CNNdualPlus":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrDual4CNN();
    	else
    		conf = getNetworkConfigurationBoth2_4CNN();
    	break;
    	
    }
    
    return conf;
	
}


public static ComputationGraph doExperimentTesting(String trainingDataPath, String featuresAddPath, String labelsPath) throws Exception{
    	
    	
	    MultiDataSetIterator trainIter = getTrainData(trainingDataPath, featuresAddPath, labelsPath);
    	
	    ComputationGraphConfiguration conf = getNNConfig();
        
        
        int listenerFrequency = 1;
    	ComputationGraph model = new ComputationGraph(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(listenerFrequency));
        for( int i=0; i<nEpochs; i++ ) {
        	System.out.println("Epoch: " + i);
            model.fit(trainIter);
            trainIter.reset();
        }
        System.out.println("Training Finished!!!");
        
        return model;
        
    }
    
    
    public static void doExperimentTraining(String trainingDataPath, String featuresAddPath, String labelsPath) throws Exception{
    	
    	MultiDataSetIterator trainIter = getTrainData(trainingDataPath, featuresAddPath, labelsPath);
    	
	    ComputationGraphConfiguration conf = getNNConfig();
        
        int listenerFrequency = 1;
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        
    	ComputationGraph model = new ComputationGraph(conf);
    	model.setListeners(new StatsListener(statsStorage, listenerFrequency));
        uiServer.attach(statsStorage);
        model.init();
        
        //model.setListeners(new ScoreIterationListener(listenerFrequency));
        
        for( int i=0; i<nEpochs; i++ ) {
        	System.out.println("Epoch: " + i);
        	model.fit(trainIter);
            trainIter.reset();
        }
        System.out.println("Training Finished!!!");
    }
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationAlLoro2Class() {
        final int cnnLayerFeatureMaps = 100;
        
        INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn3", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                .kernelSize(window+3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(new LossMCXENT(weightsArray))
                .activation(Activation.SOFTMAX)
                .nIn(3 * cnnLayerFeatureMaps + nAddFeat)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "globalPool", "input2")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.feedForward(nAddFeat))
            .build();
    }
    
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                //.dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                //.dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(2 * 3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(2 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2Simple() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(2 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "globalPool_w2v", "globalPool_feat")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2_4CNN() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn2_w2v", new ConvolutionLayer.Builder()
                    .kernelSize(2, width)
                    .stride(1, width)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(4, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(5, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn2_w2v", "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn2_feat", new ConvolutionLayer.Builder()
                    .kernelSize(2, 30)
                    .stride(1, 30)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input2")
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(4, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(5, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            
            .addVertex("merge_feat", new MergeVertex(), "cnn2_feat", "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(2 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		//.dropOut(0.5)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(2 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2Dual() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(2 * 3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(2 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2Regr() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            .activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrDual() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            .activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3, width)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,2), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrDual4CNN() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            .activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn2_w2v", new ConvolutionLayer.Builder()
                    .kernelSize(window, width)
                    .stride(1, width)
                    .nIn(2)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3, width)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn2_w2v", "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.25)
                .build(), "merge_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.1)
                .build(), "merge_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,2), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2Regr4CNN() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            .activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn2_w2v", new ConvolutionLayer.Builder()
                    .kernelSize(window, width)
                    .stride(1, width)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn2_w2v", "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2Regr4CNN_Embedding(int vocasize) {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            .activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("embedding", new EmbeddingLayer.Builder()
            		.nIn(vocasize+1)
            		.nOut(width)
            		.build(),"input")
            .addLayer("cnn2_w2v", new ConvolutionLayer.Builder()
                    .kernelSize(window, width)
                    .stride(1, width)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "embedding")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "embedding")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "embedding")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn2_w2v", "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.feedForward(height), InputType.convolutionalFlat(7,30,nChannels))
            .inputPreProcessor("cnn2_w2v", new FeedForwardToCnnPreProcessor(height, width, nChannels))
            .inputPreProcessor("cnn3_w2v", new FeedForwardToCnnPreProcessor(height, width, nChannels))
            .inputPreProcessor("cnn4_w2v", new FeedForwardToCnnPreProcessor(height, width, nChannels))
            .inputPreProcessor("cnn5_w2v", new FeedForwardToCnnPreProcessor(height, width, nChannels))
            .build();
            
    }
    
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationAlLoro2Regre() {
        final int cnnLayerFeatureMaps = 100;
        
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn3", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                .kernelSize(window+3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(3 * cnnLayerFeatureMaps + nAddFeat)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "globalPool", "input2")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.feedForward(nAddFeat))
            .build();
    }
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationWindowClass() {
        final int cnnLayerFeatureMaps = 100;
        
        INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        
        return new NeuralNetConfiguration.Builder()
        	//.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input")
            .addLayer("cnn3", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                    .kernelSize(window+3, width)
                    .stride(1, width)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input")
            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(new LossMCXENT(weightsArray))
                //.lossFunction(new LossNegativeLogLikelihood(weightsArray))
                .activation(Activation.SOFTMAX)
                .nIn(3 * cnnLayerFeatureMaps)
                .nOut(classes)    //2 classes: positive or negative
                .build(), "globalPool")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels))
            .build();
    }
    
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationWindowRegre() {
        final int cnnLayerFeatureMaps = 100;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input")
            .addLayer("cnn3", new ConvolutionLayer.Builder()
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                    .kernelSize(window+3, width)
                    .stride(1, width)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input")
            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(3 * cnnLayerFeatureMaps)
                .nOut(classes)    //2 classes: positive or negative
                .build(), "globalPool")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels))
            .build();
    }
    

    public static int countLines(String filename) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(filename));
        try {
            byte[] c = new byte[1024];
            int count = 0;
            int readChars = 0;
            boolean empty = true;
            while ((readChars = is.read(c)) != -1) {
                empty = false;
                for (int i = 0; i < readChars; ++i) {
                    if (c[i] == '\n') {
                        ++count;
                    }
                }
            }
            return (count == 0 && !empty) ? 1 : count;
        } finally {
            is.close();
        }
    }


}

