package controller;

import java.awt.BorderLayout;
import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import neuralnetwork.CustomLayer;
import play.mvc.WebSocket.In;
import utils.Constants;
import utils._utils;
import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.j48.NBTreeClassifierTree;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class Main {
	

    private static Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
//        int batchSize = 64; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
//        int numEpochs = 15; // number of epochs to perform
        double rate = 0.0015; // learning rate

        //Get the DataSetIterators:

        
        DataSource source = new DataSource("mnist.arff");
		Instances dataset = source.getDataSet();
		System.out.println(dataset.size());
		String[] options;
		System.out.println(dataset.attribute(783).isNumeric());
		System.out.println(dataset.numAttributes());
		NumericToNominal convert = new NumericToNominal();
		options = new String[2];
		options[0] = "-R";
		System.out.println(dataset.attribute(783).isNumeric());
		options[1] = "" + (dataset.numAttributes()); // range of variables to make numeric 
		convert.setOptions(options);
		convert.setInputFormat(dataset);
		dataset = weka.filters.Filter.useFilter(dataset, convert);
		System.out.println(dataset.attribute(783).isNumeric());
		dataset.setClassIndex(dataset.numAttributes() - 1);
		

		int batchSize = 100; // Iris data set: 150 examples total. We are loading all of them into one
		int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
		int testSize = dataset.numInstances() - trainSize;
		dataset.randomize(new java.util.Random());	// randomize instance order before splitting dataset
		Instances training = new Instances(dataset, 0, trainSize);
		Instances test = new Instances(dataset, trainSize, testSize);

		DataSet trainingData = _utils.instancesToDataSet(training);
		DataSet testData = _utils.instancesToDataSet(test);			
		int  batchNum = trainingData.numExamples() / batchSize;

		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);
        

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            .activation(Activation.SIGMOID)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(rate, 0.98))
            .l2(rate * 0.005) // regularize learning model
            .list()
            .layer(0, new DenseLayer.Builder() //create the first input layer.
                    .nIn(numRows * numColumns)
                    .nOut(500)
                    .build())
            .layer(1, new DenseLayer.Builder() //create the second input layer
                    .nIn(500)
                    .nOut(100)
                    .build())
            .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .activation(Activation.SOFTMAX)
                    .nIn(100)
                    .nOut(outputNum)
                    .build())
            .pretrain(false).backprop(true) //use backpropagation to adjust weights
            .build();

        Constants.model = new MultiLayerNetwork(conf);
        Constants.model.init();
        Constants.model.setListeners(new ScoreIterationListener(5));  //print the score with every iteration

        log.info("Train model....");
        for( int i=0; i<10; i++ ){
        	log.info("Epoch " + i);
       	for  ( int  b = 0 ; b < batchNum ; b ++){
        		DataSet set = getBatchTrainSet(b, batchSize, trainingData, training);
//        	while( mnistTrain.hasNext()){
        		Constants.model.fit(set);
        	}
        		
        				
//        	}
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
//       while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = Constants.model.output(testData.getFeatures()); //get the networks prediction
            System.out.println(output.shapeInfoToString());
            eval.eval(testData.getLabels(), output); //check the prediction against the true class
//        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }
    
	private static DataSet getBatchTrainSet(int batchNumber , int batchRate,  	DataSet trainSet , Instances training) {
		
		INDArray features = trainSet.getFeatures();
		INDArray labels = trainSet.getLabels();
		int start =  batchNumber * batchRate;
		int end =  (batchNumber + 1) * batchRate;
		
		INDArray batchTrain_features = features.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
		INDArray batchTrain_labels = labels.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
		
		DataSet set = new DataSet(batchTrain_features, batchTrain_labels);
		 List<Instance> list = training.subList(start, end);
		 double [] labels_list = new double[list.size()];
		 for( int i =0 ; i< list.size() ; i++)
			 labels_list[i] = list.get(i).classValue(); 
//		 System.out.println(labels.length());
//		 System.out.println(batchTrain_labels.shapeInfoToString());
//		 System.out.println(batchTrain_labels);
//		 System.out.println(labels_list[labels_list.length - 1]);
		Constants.trainInstancesLabel = Nd4j.create(labels_list).transpose();
//		System.out.println(Constants.trainInstancesLabel.shapeInfoToString());
//		System.out.println(set.numExamples());
		
		return set;
		
	}
    
}
