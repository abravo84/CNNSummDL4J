package edu.upf.taln.cnnsumm;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.*;
import java.util.*;

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
    public static boolean trainMode = false;
    public static int nClusters = 0;
    public static String expe="";
    
    public static int nAddFeat = 19;
    
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
        nFilters = Integer.parseInt(args[9]);
        expe = args[10];
        algorithType = args[11];
        classes = Integer.parseInt(args[12]);
        inputMode = args[13];
        window = Integer.parseInt(args[14]);
        
        
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
        String citationsPath = args[20];
        nClusters = Integer.parseInt(args[21]);
        String citationPrefix = args[22];
        
        
        nAddFeat = Integer.parseInt(args[23]);
        
         
       String trainModeStr = args[24];
       if (trainModeStr.toLowerCase().contains("t")) 
    	   trainMode=true;
        
        
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
        System.out.println("nAddFeat: " + nAddFeat);
        System.out.println("-------------------------------------");
        
        
        //String featuresType = args[9]; // token_vec, token_google_vec, acl_vec
        
        String experimentNumber = args[10];//"_01";
        
        String reg = "reg";
        if (classes>1)
        	reg="class";
        
        experimentNumber = experimentNumber + "_a" + algorithType +  "_i" +inputMode +  "_w" +window +  "_e" +nEpochs +  "_lr" + learningRate +  "_" + reg;
        ArrayList<String> optionList=new ArrayList<String>();
        //optionList.add("human");
        //optionList.add("community");
        optionList.add("abstract");
        Nd4j.getMemoryManager().setAutoGcWindow(15000);
        
        //String trainingDataPath = "/home/upf/corpora/SciSUM-2017-arffs-and-models/training_sets_ab/ALL-DOCS-2016-TRAIN-27-FILES/token_acl_vec_human.txt";
        
        
        if (trainMode && testMode) {
        	ComputationGraph model = doExperimentTesting(trainingDataPath, featuresAddPath, citationsPath,labelsPath);
	    	File locationToSave = new File("/homedtic/abravo/model_"+expe+".zip");
	        ModelSerializer.writeModel(model, locationToSave, false); 
	        predictionCG(model, testDataPath, vecPrefix, featPrefix, citationPrefix, resultPath, experimentNumber);
	        System.exit(0);
        }
        
        if (trainMode) {        	
        	ComputationGraph model = doExperimentTesting(trainingDataPath, featuresAddPath, citationsPath,labelsPath);
	    	File locationToSave = new File("/homedtic/abravo/model_"+expe+".zip");
	        ModelSerializer.writeModel(model, locationToSave, false);        	
        }
        
        if(testMode) {
        	if (!trainMode) {
        		ComputationGraph model2 = doExperimentTesting(trainingDataPath, featuresAddPath, citationsPath,labelsPath);
        	}
        	ComputationGraph model = ModelSerializer.restoreComputationGraph("/homedtic/abravo/model_"+expe+".zip", false);
        	predictionCG(model, testDataPath, vecPrefix, featPrefix, citationPrefix, resultPath, experimentNumber);        	
        }
        
        System.exit(0);
        
        if (testMode){
        	ComputationGraph model = doExperimentTesting(trainingDataPath, featuresAddPath, citationsPath,labelsPath);
        	File locationToSave = new File("/homedtic/abravo/model_"+expe+".zip");
            ModelSerializer.writeModel(model, locationToSave, false);
            //System.exit(0);
            
        	
        	
        	//ComputationGraph model = ModelSerializer.restoreComputationGraph("/homedtic/abravo/model_"+expe+".zip", false);
        	predictionCG(model, testDataPath, vecPrefix, featPrefix, citationPrefix, resultPath, experimentNumber);
        	
        }
        else {
        	//doExperimentTraining(trainingDataPath, featuresAddPath, citationsPath, labelsPath);
        	ComputationGraph model = doExperimentTesting(trainingDataPath, featuresAddPath, citationsPath,labelsPath);
        	File locationToSave = new File("/homedtic/abravo/model_"+expe+".zip");
            ModelSerializer.writeModel(model, locationToSave, false);
        }
    }
    
    public static void predictionCG(ComputationGraph model, String testDataPath, String vecPrefix, String featPrefix, String citationPrefix, String testPathFinal, String experimentNumber) throws IOException, InterruptedException {
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
            String citationsPath = testDataPath + File.separator + citationPrefix + docID + ".csv";
            
            
            System.out.println("w2vPath FILE: " + w2vPath + "   " + countLines(w2vPath));
            System.out.println("featuresAddPath FILE: " + featuresAddPath + "   " + countLines(featuresAddPath));
            System.out.println("citationsPath FILE: " + citationsPath + "   " + countLines(citationsPath));
            		
            
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
	    		.addInput("addFeat",1,nAddFeat*7)
	    		.build();
	        	}else if (inputMode.equals("2db")) {
	        		RecordReader addFeatRecord = new CSVRecordReader(0, ',');
		        	addFeatRecord.initialize(new FileSplit(new File(featuresAddPath)));
		    		testIter =new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
	        	    		.addReader("word2vec", word2vecRecord)
	        	    		.addReader("", addFeatRecord)
	        	    		.addInput("word2vec", 1, totalRow*2)
	        	    		.addInput("addFeat",1,nAddFeat*7)
	        	    		.build();
	            	}else if (inputMode.equals("3")) {
	    		RecordReader addFeatRecord = new CSVRecordReader(0, ',');
	        	addFeatRecord.initialize(new FileSplit(new File(featuresAddPath)));
	        	RecordReader citationsRecord = new CSVRecordReader(0, ',');
	        	citationsRecord.initialize(new FileSplit(new File(citationsPath)));
	        	testIter = new RecordReaderMultiDataSetIterator.Builder(testBatch)
	        			.addReader("word2vec", word2vecRecord)
    		    		.addReader("addFeat", addFeatRecord)
    		    		.addReader("cit2vec", citationsRecord)
    		    		.addInput("word2vec", 1, totalRow)
    		    		.addInput("addFeat",1,nAddFeat*7)
    		    		.addInput("cit2vec",1,nClusters*300)
    		    		.build();
	    	}else if (inputMode.equals("3d")) {
	    		RecordReader addFeatRecord = new CSVRecordReader(0, ',');
	        	addFeatRecord.initialize(new FileSplit(new File(featuresAddPath)));
	        	RecordReader citationsRecord = new CSVRecordReader(0, ',');
	        	citationsRecord.initialize(new FileSplit(new File(citationsPath)));
	    		testIter = new RecordReaderMultiDataSetIterator.Builder(testBatch)
	    				.addReader("word2vec", word2vecRecord)
    		    		.addReader("addFeat", addFeatRecord)
    		    		.addReader("cit2vec", citationsRecord)
    		    		.addInput("word2vec", 1, totalRow*2)
    		    		.addInput("addFeat",1,nAddFeat*7)
    		    		.addInput("cit2vec",1,nClusters*300)
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

public static MultiDataSetIterator getTrainData(String trainingDataPath, String featuresAddPath, String citationsPath, String labelsPath) throws IOException, InterruptedException {
	//trainingDataPath = featuresPath + File.separator + "acl_vec_"+height+".txt";
	int totalRow = height*width;
	RecordReader word2vecRecord = new CSVRecordReader(0, ',');
	word2vecRecord.initialize(new FileSplit(new File(trainingDataPath)));
	
	RecordReader addFeatRecord = new CSVRecordReader(0, ',');
	addFeatRecord.initialize(new FileSplit(new File(featuresAddPath)));
	
	RecordReader citationsRecord = null;
	if (citationsPath.length()>5) {
    	citationsRecord = new CSVRecordReader(0, ',');
    	citationsRecord.initialize(new FileSplit(new File(citationsPath)));
	}
	
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
	    		.addInput("addFeat",1,nAddFeat*7)
	    		.addOutput("labels", 1, 1) 
	    		.build();
    	}else if (inputMode.equals("2d")) {
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
    	    		.addReader("word2vec", word2vecRecord)
    	    		.addReader("addFeat", addFeatRecord)
    	    		.addReader("labels", labelsRecord)
    	    		.addInput("word2vec", 1, totalRow*2)
    	    		.addInput("addFeat",1,nAddFeat*7)
    	    		.addOutput("labels", 1, 1) 
    	    		.build();
        	}else if (inputMode.equals("2db")) {
        		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
        	    		.addReader("word2vec", word2vecRecord)
        	    		.addReader("", addFeatRecord)
        	    		.addInput("word2vec", 1, totalRow*2)
        	    		.addInput("addFeat",1,nAddFeat*7)
        	    		.addOutput("addFeat", nAddFeat*7, nAddFeat*7) 
        	    		.build();
            	}else if (inputMode.equals("2emb")) {
        		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
        	    		.addReader("word2vec", word2vecRecord)
        	    		.addReader("addFeat", addFeatRecord)
        	    		.addReader("labels", labelsRecord)
        	    		.addInput("word2vec", 1,15)
        	    		.addInput("addFeat",1,nAddFeat*7)
        	    		.addOutput("labels", 1, 1) 
        	    		.build();
            	}else if (inputMode.equals("3")) {
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
		    		.addReader("word2vec", word2vecRecord)
		    		.addReader("addFeat", addFeatRecord)
		    		.addReader("cit2vec", citationsRecord)
		    		.addReader("labels", labelsRecord)
		    		.addInput("word2vec", 1, totalRow)
		    		.addInput("addFeat",1,nAddFeat*7)
		    		.addInput("cit2vec",1,nClusters*300)
		    		.addOutput("labels", 1, 1)
		    		.build();
    	}else if (inputMode.equals("3d")) {
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
		    		.addReader("word2vec", word2vecRecord)
		    		.addReader("addFeat", addFeatRecord)
		    		.addReader("cit2vec", citationsRecord)
		    		.addReader("labels", labelsRecord)
		    		.addInput("word2vec", 1, totalRow*2)
		    		.addInput("addFeat",1,nAddFeat*7)
		    		.addInput("cit2vec",1,nClusters*300)
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
    	}else if (inputMode.equals("2")){
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
    	    		.addReader("word2vec", word2vecRecord)
    	    		.addReader("addFeat", addFeatRecord)
    	    		.addReader("labels", labelsRecord)
    	    		.addInput("word2vec", 1, totalRow*2)
    	    		.addInput("addFeat",1,30*7)
    	    		.addOutputOneHot("labels", 1, classes)
    	    		.build();
        	}else if (inputMode.equals("3")) {
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
		    		.addReader("word2vec", word2vecRecord)
		    		.addReader("addFeat", addFeatRecord)
		    		.addReader("cit2vec", citationsRecord)
		    		.addReader("labels", labelsRecord)
		    		.addInput("word2vec", 1, totalRow)
		    		.addInput("addFeat",1,30*7)
		    		.addInput("cit2vec",1,nClusters*300)
		    		.addOutputOneHot("labels", 1, classes)
		    		.build();
    	}else if (inputMode.equals("3d")) {
    		trainIter = new RecordReaderMultiDataSetIterator.Builder(miniBatchSize)
		    		.addReader("word2vec", word2vecRecord)
		    		.addReader("addFeat", addFeatRecord)
		    		.addReader("cit2vec", citationsRecord)
		    		.addReader("labels", labelsRecord)
		    		.addInput("word2vec", 1, totalRow*2)
		    		.addInput("addFeat",1,30*7)
		    		.addInput("cit2vec",1,nClusters*300)
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
    case "oldCNN":
    	System.out.println("--> oldCNN");
    	if (classes==1)
    		conf = getNetworkConfigurationWindowRegre();
    	else
    		conf = getNetworkConfigurationWindowClass();
    	break;
    	
    case "newCNN":
    	System.out.println("--> newCNN");
    	if (classes==1)
    		conf = getNetworkConfigurationAlLoro2Regre();
    	else
    		conf = getNetworkConfigurationAlLoro2Class();
    	break;
    	
    case "oldCNN_B":
    	conf = getNetworkConfigurationWindowClassB();
    	break;
    case "alg":
    	conf = getNetworkConfigurationBoth();
    	break;
    case "alg2":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2Regr();
    	else
    		conf = getNetworkConfigurationBoth2();
    	break;
    case "alg2simple":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2Regr();
    	else
    		conf = getNetworkConfigurationBoth2Simple();
    	break;	
    case "alg2_4cnn":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2Regr4CNN();
    	else
    		conf = getNetworkConfigurationBoth2_4CNN();
    	break;	
    case "alg2dual":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrDual();
    	else
    		conf = getNetworkConfigurationBoth2Dual();
    	break;
    	
    case "alg2dual_4cnn":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrDual4CNN();
    	else
    		conf = getNetworkConfigurationBoth2_4CNN();
    	break;
    	
    	
    case "alg2dual_4cnn_a":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrDual4CNN_a();
    	
    case "alg2dual_4cnn_b":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrDual4CNN_b();
    	
    	
    case "alg2dual_4cnn_1fc":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrDual4CNN_1FC();
    	
    case "alg2dual_4cnn_all":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrDual4CNNAll();
    	else
    		conf = getNetworkConfigurationBoth2_4CNN();
    	break;
    	
    case "alg2l":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrLight();
    	else
    		conf = getNetworkConfigurationBoth2Light();
    	break;
    	
    case "alg2lnost":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2RegrLight();
    	else
    		conf = getNetworkConfigurationBoth2LightNonStatic();
    	break;
    		
    case "alg3dual_4cnn_a":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth3RegrDual4CNN_a();	
    	
    case "alg2B":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth2BRegr();
    	else
    		conf = getNetworkConfigurationBoth2();
    	break;
    case "alg3":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth3RegrLight();
    	else
    		conf = getNetworkConfigurationBoth3();
    	break;
    	
    	
    	
    case "alg3B":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth3BRegr();
    	else
    		conf = getNetworkConfigurationBoth3B();
    	break;
    	
    case "alg3C":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth3BRegr();
    	else
    		conf = getNetworkConfigurationBoth3C();
    	break;	
        	
    case "alg3dual":
    	if (classes==1)
    		conf = getNetworkConfigurationBoth3DualRegr();
    	else
    		conf = getNetworkConfigurationBoth3Dual();
    	break;
    
    case "m3":
    	if (classes==1)
    		conf = getNetworkConfiguration3MultOutRegr();
    	else
    		conf = getNetworkConfigurationBoth3();
    	break;
    }
    
    
    return conf;
	
}


public static ComputationGraph doExperimentTesting(String trainingDataPath, String featuresAddPath, String citationsPath, String labelsPath) throws Exception{
    	
    	
	    MultiDataSetIterator trainIter = getTrainData(trainingDataPath, featuresAddPath, citationsPath, labelsPath);
    	
	    if (trainMode) {
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
	    return null;
        
    }
    
    
    public static void doExperimentTraining(String trainingDataPath, String featuresAddPath, String citationsPath, String labelsPath) throws Exception{
    	
    	MultiDataSetIterator trainIter = getTrainData(trainingDataPath, featuresAddPath, citationsPath, labelsPath);
    	
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
        
        File locationToSave = new File("/homedtic/abravo/model_"+expe+".zip");
        ModelSerializer.writeModel(model, locationToSave, false);
        
        
        System.out.println("Training Finished!!!");
    }
    
    
    
    

    private static MultiLayerConfiguration getNetworkConfiguration() {
        return new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            //.regularization(true).l2(0.0005)
            .regularization(true).l2(1e-4)
            .learningRate(learningRate)//.biasLearningRate(0.02)
            //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, width)
                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                .name("hzvt1")
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(26)
                .activation(Activation.RELU)//.activation("identity")
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(classes)
                //.activation(Activation.SOFTMAX)
                .activation(Activation.IDENTITY)
                .build())
            .setInputType(InputType.convolutionalFlat(height,width,nChannels))
            .backprop(true).pretrain(false).build();
    }
    
    private static MultiLayerConfiguration getNetworkConfigurationConvu() {
    	
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(false).l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .learningRate(0.0001) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.9))
                .list()
                .layer(0, convInit("cnn1", nChannels, 100,  new int[]{3, width}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,1}))
                //.layer(2, conv5x5("cnn2", 100, new int[]{5, 5}, new int[]{1, 1}, 0))
                //.layer(3, maxPool("maxool2", new int[]{2,2}))
                .layer(2, new DenseLayer.Builder().nOut(100).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(classes)
                    .activation(Activation.SOFTMAX)
                    .build())
                .backprop(true).pretrain(false)
                .setInputType(InputType.convolutionalFlat(height, width, nChannels))
                .build();
    
    	
    	return conf;
    }
    
    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }
    private static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private static SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }
    private static MultiLayerConfiguration getNetworkConfiguration1() {
    	INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        ConvolutionLayer layer0 = new ConvolutionLayer.Builder(2,width)
            .nIn(1)
            .nOut(16)
            .stride(1,1)
            //.padding(2,2)
            .weightInit(WeightInit.XAVIER)
            .name("First convolution layer")
            .activation(Activation.RELU)
            .build();

        SubsamplingLayer layer1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2,2)
            .stride(2,2)
            .name("First subsampling layer")
            .build();

        ConvolutionLayer layer2 = new ConvolutionLayer.Builder(5,5)
            .nOut(20)
            .stride(1,1)
            .padding(2,2)
            .weightInit(WeightInit.XAVIER)
            .name("Second convolution layer")
            .activation(Activation.RELU)
            .build();

        SubsamplingLayer layer3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2,2)
            .stride(2,2)
            .name("Second subsampling layer")
            .build();

        ConvolutionLayer layer4 = new ConvolutionLayer.Builder(5,5)
            .nOut(20)
            .stride(1,1)
            .padding(2,2)
            .weightInit(WeightInit.XAVIER)
            .name("Third convolution layer")
            .activation(Activation.RELU)
            .build();

        SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2,2)
            .stride(2,2)
            .name("Third subsampling layer")
            .build();

        OutputLayer layer6 = new OutputLayer.Builder()
            .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            //.lossFunction(new LossNegativeLogLikelihood(weightsArray))
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .name("Output")
            .nOut(classes)
            .build();
        
        
        
        
        
        return new NeuralNetConfiguration.Builder()
            .seed(12345)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .regularization(true)
            .l2(0.0004)
            .updater(Updater.NESTEROVS)
            .momentum(0.9)
            .list()
            .layer(0, layer0)
            .layer(1, layer1)
            //.layer(2, layer2)
            //.layer(3, layer3)
            //.layer(4, layer4)
            //.layer(5, layer5)
            .layer(2, layer6)
            .pretrain(false)
            .backprop(true)
            .setInputType(InputType.convolutionalFlat(height,width,nChannels))
            .build();
    }

    private static MultiLayerConfiguration getNetworkConfiguration1_old() {

        ConvolutionLayer layer0 = new ConvolutionLayer.Builder(5,5)
            .nIn(1)
            .nOut(16)
            .stride(1,1)
            .padding(2,2)
            .weightInit(WeightInit.XAVIER)
            .name("First convolution layer")
            .activation(Activation.RELU)
            .build();

        SubsamplingLayer layer1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2,2)
            .stride(2,2)
            .name("First subsampling layer")
            .build();

        ConvolutionLayer layer2 = new ConvolutionLayer.Builder(5,5)
            .nOut(20)
            .stride(1,1)
            .padding(2,2)
            .weightInit(WeightInit.XAVIER)
            .name("Second convolution layer")
            .activation(Activation.RELU)
            .build();

        SubsamplingLayer layer3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2,2)
            .stride(2,2)
            .name("Second subsampling layer")
            .build();

        ConvolutionLayer layer4 = new ConvolutionLayer.Builder(5,5)
            .nOut(20)
            .stride(1,1)
            .padding(2,2)
            .weightInit(WeightInit.XAVIER)
            .name("Third convolution layer")
            .activation(Activation.RELU)
            .build();

        SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2,2)
            .stride(2,2)
            .name("Third subsampling layer")
            .build();

        OutputLayer layer6 = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .name("Output")
            .nOut(classes)
            .build();

        return new NeuralNetConfiguration.Builder()
            .seed(12345)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .regularization(true)
            .l2(0.0004)
            .updater(Updater.NESTEROVS)
            .momentum(0.9)
            .list()
            .layer(0, layer0)
            .layer(1, layer1)
            .layer(2, layer2)
            .layer(3, layer3)
            .layer(4, layer4)
            .layer(5, layer5)
            .layer(6, layer6)
            .pretrain(false)
            .backprop(true)
            .setInputType(InputType.convolutionalFlat(height,width,nChannels))
            .build();
    }

    private static ComputationGraphConfiguration getNetworkConfigurationAlLoro() {
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
                .kernelSize(3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(4, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                .kernelSize(5, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge", new MergeVertex(), "cnn3", "cnn4", "cnn5")      //Perform depth concatenation
            
            //.addLayer("out", <layer>, "globalPool", "input2") 
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge")
            .addVertex("merge2", new MergeVertex(), "globalPool", "input2")
            //.addLayer("ffn1", new DenseLayer.Builder()
            //		.nOut(3 * cnnLayerFeatureMaps + nAddFeat)
            //		.biasInit(1)
            //		.dropOut(0.5)
            //		.dist(new GaussianDistribution(0, 0.005))
            //		.build(), "merge2")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(new LossMCXENT(weightsArray))
                .activation(Activation.SOFTMAX)
                .nIn(3 * cnnLayerFeatureMaps + nAddFeat)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "merge2")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.feedForward(nAddFeat))
            .build();
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
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth() {
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
            
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(new LossMCXENT(weightsArray))
                .activation(Activation.SOFTMAX)
                .nIn(2 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "globalPool_w2v", "globalPool_feat")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
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
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2Light() {
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
            		//.biasInit(bias)
            		//.dropOut(0.5)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2LightNonStatic() {
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
            
            .addLayer("w2v", new DenseLayer.Builder()
            		.nIn(height*width*nChannels)
            		.nOut(height*width*nChannels)
            		//.biasInit(bias)
            		//.dropOut(0.5)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "input")
            
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "w2v")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "w2v")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "w2v")
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
            		//.biasInit(bias)
            		//.dropOut(0.5)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .inputPreProcessor("cnn3_w2v", new FeedForwardToCnnPreProcessor(height,width,nChannels))
            .inputPreProcessor("cnn4_w2v", new FeedForwardToCnnPreProcessor(height,width,nChannels))
            .inputPreProcessor("cnn5_w2v", new FeedForwardToCnnPreProcessor(height,width,nChannels))
            
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2Dual() {
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
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3() {
        final int cnnLayerFeatureMaps = nFilters;
        
        INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.1, 0.1, 1.0});
        int nAddFeat = 30;
        
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
            .addInputs("input", "input2", "input3")
            
            //1st CNN
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            //2nd CNN
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            //3th CNN
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_cit")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(3 * 3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.activation(Activation.LEAKYRELU)
            		.build(), "globalPool_w2v", "globalPool_feat", "globalPool_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(3 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3C() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.1, 0.1, 1.0});
        int nAddFeat = 30;
        
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
            .addInputs("input", "input2", "input3")
            
            //1st CNN
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("dense_layer_w2v", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"globalPool_w2v")
            
            //2nd CNN
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            
            .addLayer("dense_layer_feat", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"globalPool_feat")
            
            //3th CNN
            .addLayer("merge_cit", new DenseLayer.Builder()
                .activation(Activation.LEAKYRELU)
                .nIn(300*5)
                .nOut(cnnLayerFeatureMaps*3)
                .build(), "input3")
            
            .addLayer("dense_layer_cit", new DenseLayer.Builder()
            		.nIn(cnnLayerFeatureMaps*3)
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"merge_cit")
            
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(3 * 3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.activation(Activation.LEAKYRELU)
            		.build(), "dense_layer_w2v", "dense_layer_feat", "dense_layer_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(3 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3Dual() {
        final int cnnLayerFeatureMaps = nFilters;
        
        INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.3, 1.0});
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	//.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.RELU)
            .activation(Activation.LEAKYRELU)
            .updater(Updater.ADAM)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2", "input3")
            
            //1st CNN
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.05)
                .build(), "merge_w2v")
            
            //2nd CNN
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            //3th CNN
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_cit")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(12 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.activation(Activation.LEAKYRELU)
            		.build(), "globalPool_w2v", "globalPool_feat", "globalPool_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(12 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,2), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3B() {
        final int cnnLayerFeatureMaps = nFilters;
        
        INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.1, 0.1, 1.0});
        int nAddFeat = 30;
        
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
            .addInputs("input", "input2", "input3")
            
            //1st CNN
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+1-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("dense_layer_w2v", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"globalPool_w2v")
            
            //2nd CNN
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("dense_layer_feat", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"globalPool_feat")
            
            //3th CNN
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(1, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, width)
                //.activation(Activation.LEAKYRELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_cit")
            
            .addLayer("dense_layer_cit", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"globalPool_cit")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(3 * (cnnLayerFeatureMaps/5))
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.activation(Activation.LEAKYRELU)
            		.build(), "dense_layer_w2v", "dense_layer_feat", "dense_layer_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(3 *(cnnLayerFeatureMaps/5))
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,width,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3BRegr() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            //.activation(Activation.TANH)
        	//.activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            //.updater(Updater.SGD)
            //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2", "input3")
            //1st CNN
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
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
            
            .addLayer("dense_layer_w2v", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"globalPool_w2v")
            
            //2nd CNN
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            .addLayer("dense_layer_feat", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"globalPool_feat")
            
            //3th CNN
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(1, width)
                //.activation(Activation.SOFTSIGN)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, width)
                //.activation(Activation.SOFTSIGN)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, width)
                //.activation(Activation.SOFTSIGN)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.05)
                .build(), "merge_cit")
            
            
            .addLayer("dense_layer_cit", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps/5)
            		//.activation(Activation.SOFTSIGN)
            		.build(),"globalPool_cit")
            
            
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(3 * (cnnLayerFeatureMaps/5))
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "dense_layer_w2v", "dense_layer_feat", "dense_layer_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MSE)
                //.activation(Activation.TANH)
                .activation(Activation.IDENTITY)
                .nIn(3 * (cnnLayerFeatureMaps/5))
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3RegrLight() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            //.activation(Activation.TANH)
        	//.activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            //.updater(Updater.SGD)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2", "input3")
            //1st CNN
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
            	.activation(Activation.RELU)
                .kernelSize(window, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            /*
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
            	.activation(Activation.RELU)
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
            	.activation(Activation.RELU)
                .kernelSize(window+3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge_w2v", new MergeVertex(), "cnn3_w2v", "cnn4_w2v", "cnn5_w2v")      //Perform depth concatenation*/
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "cnn3_w2v")
            
            //2nd CNN
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .activation(Activation.RELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            /*.addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .activation(Activation.RELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .activation(Activation.RELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation*/
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "cnn3_feat")
            
            //3th CNN
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(window, width)
                .activation(Activation.RELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            /*.addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, 300)
                .activation(Activation.RELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, 300)
                .activation(Activation.RELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation*/
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "cnn1_cit")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat", "globalPool_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MSE)
                //.activation(Activation.IDENTITY)
                .activation(Activation.SIGMOID)
                .nIn(3  * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3Regr() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            //.activation(Activation.TANH)
        	//.activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            //.updater(Updater.SGD)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2", "input3")
            //1st CNN
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
            	.activation(Activation.RELU)
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
            	.activation(Activation.RELU)
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
            	.activation(Activation.RELU)
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
            
            //2nd CNN
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .activation(Activation.RELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .activation(Activation.RELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .activation(Activation.RELU)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            //3th CNN
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(1, 300)
                .activation(Activation.RELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, 300)
                .activation(Activation.RELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, 300)
                .activation(Activation.RELU)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_cit")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(3 * 3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat", "globalPool_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                //.activation(Activation.SIGMOID)
                .nIn(3 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3DualRegr() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            //.activation(Activation.TANH)
        	//.activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            //.updater(Updater.SGD)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2", "input3")
            
            //////////////////////////////////////////
            // 1st CNN
            //////////////////////////////////////////
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(2)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
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
            
            //////////////////////////////////////////
            // 2nd CNN
            //////////////////////////////////////////
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            //////////////////////////////////////////
            // 3th CNN
            //////////////////////////////////////////
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(1, 300)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, 300)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, 300)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.05)
                .build(), "merge_cit")
            
            //////////////////////////////////////////
            // FULLY
            //////////////////////////////////////////
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut((3*3 +3) * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat", "globalPool_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.TANH)
                //.activation(Activation.SIGMOID)
                .nIn((3*3 +3) * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,2), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    
    private static ComputationGraphConfiguration getNetworkConfiguration3MultOutRegr() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            //.activation(Activation.RELU)
        	//.activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            //.updater(Updater.SGD)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2", "input3")
            //1st CNN
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
                .kernelSize(window+1, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
                .kernelSize(window+2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
            	//.activation(Activation.SOFTSIGN)
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
            
            //2nd CNN
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 30)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addVertex("merge_feat", new MergeVertex(), "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_feat")
            
            //3th CNN
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(1, 300)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, 300)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, 300)
                //.activation(Activation.SOFTSIGN)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_cit")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(2*3 * 3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat", "globalPool_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MSE)
                //.activation(Activation.RELU)
                .activation(Activation.SIGMOID)
                .nIn(2*3 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .addLayer("out2", new OutputLayer.Builder()
                	//.lossFunction(new LossMCXENT(weightsArray))
                	.lossFunction(LossFunctions.LossFunction.MSE)
                    //.activation(Activation.RELU)
                    .activation(Activation.SIGMOID)
                    .nIn(3 * 3 * cnnLayerFeatureMaps)
                    .nOut(classes)    //4 classes: positive or negative
                    .build(), "fully")
            .addLayer("out3", new OutputLayer.Builder()
                	//.lossFunction(new LossMCXENT(weightsArray))
                	.lossFunction(LossFunctions.LossFunction.MSE)
                    //.activation(Activation.RELU)
                    .activation(Activation.SIGMOID)
                    .nIn(3 * 3 * cnnLayerFeatureMaps)
                    .nOut(classes)    //4 classes: positive or negative
                    .build(), "fully")
            .addLayer("out4", new OutputLayer.Builder()
                	//.lossFunction(new LossMCXENT(weightsArray))
                	.lossFunction(LossFunctions.LossFunction.MSE)
                    //.activation(Activation.RELU)
                    .activation(Activation.SIGMOID)
                    .nIn(3 * 3 * cnnLayerFeatureMaps)
                    .nOut(classes)    //4 classes: positive or negative
                    .build(), "fully")
            .setOutputs("out", "out2", "out3", "out4")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3RegrModal() {
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
            .addInputs("input", "input2", "input3")
            //1st CNN
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
            
            //2nd CNN
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
            
            //3th CNN
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                .kernelSize(1, 300)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                .kernelSize(2, 300)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                .kernelSize(3, 300)
                .stride(1, 300)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input3")
            .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
            .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "merge_cit")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(2*3 * 3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat", "globalPool_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	//.lossFunction(new LossMCXENT(weightsArray))
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(2*3 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,2), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2BRegr() {
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
            
            .addLayer("dense_layer_w2v", new DenseLayer.Builder()
            		.nOut(50)
            		.activation(Activation.LEAKYRELU)
            		.build(),"globalPool_w2v")
            
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
            
            .addLayer("dense_layer_feat", new DenseLayer.Builder()
            		.nOut(50)
            		.activation(Activation.LEAKYRELU)
            		.build(),"globalPool_feat")
            
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(2 * 50)
            		//.biasInit(bias)
            		.dropOut(0.2)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "dense_layer_w2v", "dense_layer_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(2 *50)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrOld() {
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
            		.nOut(2 * 3 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(2 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getRNN() {
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
            //.convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input")
            .addLayer("rnn", new GravesLSTM.Builder().nIn(height*width).nOut(256)
                    .activation(Activation.TANH).build(),"input")
            
            .addLayer("out", new RnnOutputLayer.Builder().activation(Activation.IDENTITY)
                    .lossFunction(LossFunctions.LossFunction.MSE).nIn(256).nOut(1).build(), "rnn")
            .setOutputs("out")
            .setInputTypes(InputType.feedForward(height*width*nChannels))
            .inputPreProcessor("rnn", new FeedForwardToRnnPreProcessor())
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
                .kernelSize(window+1, nAddFeat)
                .stride(1, nAddFeat)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, nAddFeat)
                .stride(1, nAddFeat)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, nAddFeat)
                .stride(1, nAddFeat)
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
            .setInputTypes(InputType.convolutionalFlat(height,width,2), InputType.convolutionalFlat(7,nAddFeat,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrDual4CNN_a() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
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
            .pretrain(false).backprop(true)
            .build();
    }
    
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth3RegrDual4CNN_a() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	//.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            .activation(Activation.SIGMOID)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2", "input3")
            
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
            
            
            
            
            .addLayer("cnn1_cit", new ConvolutionLayer.Builder()
                    .kernelSize(1, 300)
                    //.activation(Activation.SOFTSIGN)
                    .stride(1, 300)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input3")
                .addLayer("cnn2_cit", new ConvolutionLayer.Builder()
                    .kernelSize(2, 300)
                    //.activation(Activation.SOFTSIGN)
                    .stride(1, 300)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input3")
                .addLayer("cnn3_cit", new ConvolutionLayer.Builder()
                    .kernelSize(3, 300)
                    //.activation(Activation.SOFTSIGN)
                    .stride(1, 300)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input3")
                .addVertex("merge_cit", new MergeVertex(), "cnn1_cit", "cnn2_cit", "cnn3_cit")      //Perform depth concatenation
                .addLayer("globalPool_cit", new GlobalPoolingLayer.Builder()
                    .poolingType(PoolingType.MAX)
                    .dropOut(0.5)
                    .build(), "merge_cit")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(cnnLayerFeatureMaps)
            		//.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat", "globalPool_cit")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,2), InputType.convolutionalFlat(7,30,nChannels),InputType.convolutionalFlat(nClusters,300,nChannels))
            .backprop(true)
            .build();
        
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrDual4CNN_b() {
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
            		.activation(Activation.IDENTITY)
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
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrDual4CNN_1FC() {
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
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(2*3*cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "globalPool_w2v", "globalPool_feat")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,2), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrDual4CNNAll() {
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
                .dropOut(0.5)
                .build(), "merge_w2v")
            
            .addLayer("cnn2_feat", new ConvolutionLayer.Builder()
                    .kernelSize(window, 30)
                    .stride(1, 30)
                    .nIn(nChannels)
                    .nOut(cnnLayerFeatureMaps)
                    .build(), "input2")
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
            .addVertex("merge_feat", new MergeVertex(),"cnn2_feat", "cnn3_feat", "cnn4_feat", "cnn5_feat")      //Perform depth concatenation
            
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
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrOld1() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            //.activation(Activation.RELU)
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
                //.activation(Activation.RELU)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+2-1, width)
                .stride(1, width)
                .nIn(nChannels)
                //.activation(Activation.RELU)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window+3-1, width)
                .stride(1, width)
                .nIn(nChannels)
                //.activation(Activation.RELU)
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
                //.activation(Activation.RELU)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn4_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 30)
                .stride(1, 30)
                .nIn(nChannels)
                //.activation(Activation.RELU)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            .addLayer("cnn5_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+3, 30)
                .stride(1, 30)
                .nIn(nChannels)
                //.activation(Activation.RELU)
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
            	.lossFunction(LossFunctions.LossFunction.MSE)
                .activation(Activation.SIGMOID)
                .nIn(2 * 3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfigurationBoth2RegrLight() {
        final int cnnLayerFeatureMaps = nFilters;
        
        //INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        //int nAddFeat = 30;
        
        return new NeuralNetConfiguration.Builder()
        	.trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
        	.seed(seed)
        	.iterations(iterations)
        	.weightInit(WeightInit.XAVIER)
            //.activation(Activation.RELU)
            .updater(Updater.ADAM)
            .convolutionMode(ConvolutionMode.Same)      //This is important so we can 'stack' the results later
            .regularization(true).l2(0.0001)
            .learningRate(learningRate)
            .graphBuilder()
            .addInputs("input", "input2")
            .addLayer("cnn3_w2v", new ConvolutionLayer.Builder()
                .kernelSize(window, width)
                .stride(1, width)
                .nIn(nChannels)
                //.activation(Activation.RELU)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            
            .addLayer("globalPool_w2v", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "cnn3_w2v")
            
            .addLayer("cnn3_feat", new ConvolutionLayer.Builder()
                .kernelSize(window+1, 30)
                .stride(1, 30)
                .nIn(nChannels)
                //.activation(Activation.RELU)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input2")
            
            .addLayer("globalPool_feat", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .dropOut(0.5)
                .build(), "cnn3_feat")
            
            .addLayer("fully", new DenseLayer.Builder()
            		.nOut(2 * cnnLayerFeatureMaps)
            		//.biasInit(bias)
            		.dropOut(0.5)
            		.dist(new GaussianDistribution(0, 0.005))
            		.build(), "globalPool_w2v", "globalPool_feat")
            
            .addLayer("out", new OutputLayer.Builder()
            	.lossFunction(LossFunctions.LossFunction.MSE)
                //.activation(Activation.SIGMOID)
                .nIn(2 *cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "fully")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels), InputType.convolutionalFlat(7,30,nChannels))
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
    
    private static ComputationGraphConfiguration getNetworkConfigurationAlLoroVec() {
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
                .dropOut(0.5)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX)
                .nIn(3 * cnnLayerFeatureMaps)
                .nOut(classes)    //4 classes: positive or negative
                .build(), "globalPool")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels))
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
    
    private static ComputationGraphConfiguration getNetworkConfigurationWindowClassB() {
        final int cnnLayerFeatureMaps = nFilters;
        
        INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        
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
                .kernelSize(window+1, 1)
                .stride(1, 1)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(window+2, 1)
                .stride(1, 1)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                    .kernelSize(window+3, 1)
                    .stride(1, 1)
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
    
    private static ComputationGraphConfiguration getCNNandLSTM() {
        final int cnnLayerFeatureMaps = 100;
        int lstmLayerSize = 256;
        int tbpttLength = 35;
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
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .build(), "cnn3", "cnn4", "cnn5")
            
            
            .addLayer("lstm1", new GravesLSTM.Builder()
                    .nOut(lstmLayerSize)
                    .updater(Updater.RMSPROP)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.RELU)
                    .build(),"globalPool")
            .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX)
                    .updater(Updater.RMSPROP)
                    .nIn(lstmLayerSize)
                    .nOut(classes)
                    .build(),"lstm1")
            
            .setOutputs("out")
            .backprop(true)
            .backpropType(BackpropType.TruncatedBPTT)
            .tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false)
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfiguration345() {
        final int cnnLayerFeatureMaps = 100;
        
        INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        
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
                .kernelSize(3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(4, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn5", new ConvolutionLayer.Builder()
                    .kernelSize(5, width)
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
                .activation(Activation.SOFTMAX)
                .nIn(3 * cnnLayerFeatureMaps)
                .nOut(classes)    //2 classes: positive or negative
                .build(), "globalPool")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels))
            .build();
    }
    
    private static ComputationGraphConfiguration getNetworkConfiguration234() {
        final int cnnLayerFeatureMaps = 100;
        
        INDArray weightsArray = Nd4j.create(new double[]{0.1, 0.3, 0.5, 1.0});
        
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
            .addLayer("cnn2", new ConvolutionLayer.Builder()
                .kernelSize(2, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn3", new ConvolutionLayer.Builder()
                .kernelSize(3, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addLayer("cnn4", new ConvolutionLayer.Builder()
                .kernelSize(4, width)
                .stride(1, width)
                .nIn(nChannels)
                .nOut(cnnLayerFeatureMaps)
                .build(), "input")
            .addVertex("merge", new MergeVertex(), "cnn2", "cnn3", "cnn4")      //Perform depth concatenation
            .addLayer("globalPool", new GlobalPoolingLayer.Builder()
                .poolingType(PoolingType.MAX)
                .build(), "merge")
            .addLayer("out", new OutputLayer.Builder()
                .lossFunction(new LossMCXENT(weightsArray))
                .activation(Activation.SOFTMAX)
                .nIn(3 * cnnLayerFeatureMaps)
                .nOut(classes)    //2 classes: positive or negative
                .build(), "globalPool")
            .setOutputs("out")
            .setInputTypes(InputType.convolutionalFlat(height,width,nChannels))
            .build();
    }

    private static ConvolutionLayer getCNN() {

        return new ConvolutionLayer.Builder(5,5)
            .nIn(3)
            .nOut(16)
            .stride(1,1)
            .padding(2,2)
            .weightInit(WeightInit.XAVIER)
            .name("First convolution layer")
            .activation(Activation.RELU)
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

