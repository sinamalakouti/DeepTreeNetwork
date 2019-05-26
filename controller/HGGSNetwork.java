package controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
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
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.NDArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import neuralnetwork.CustomLayer;
import utils.Constants;
import utils._utils;
import weka.classifiers.trees.HoeffdingTree;
//import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class HGGSNetwork {

	private static Logger log = LoggerFactory.getLogger(HGGSNetwork.class);

	public static void main(String[] args) throws Exception {


		int numLinesToSkip = 0;
		char delimiter = ',';
		RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
		recordReader.initialize(new FileSplit( new File("/Users/sina/Documents/JGU_Research/ComplexNeuronsProject/datasets/HIGGS.csv")));
		
		Constants.numClasses = 2;
		int datasetSize = 11000;
		Constants.batchSize = 100;
		Constants.numBatches = datasetSize / Constants.batchSize;
		final int numInputs = 28;
		
		
		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, datasetSize,0,Constants.numClasses);
		

			DataSet allData = iterator.next();		
			allData.shuffle();
		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.7); // Use
																			// 70%
																			// of
																			// data
																			// for
																			// training
		DataSet trainingData = testAndTrain.getTrain();
		DataSet testData = testAndTrain.getTest();
		
		System.out.println(trainingData.getLabels().shapeInfoToString());
		System.out.println(trainingData.getFeatures().shapeInfoToString());

		
		log.info("Build model....");
		Constants.numberOfLayers = 2;
		Constants.numberOfNeurons = 40;
		// org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(6)

				.trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
				.weightInit(WeightInit.XAVIER).updater(new Sgd(0.1)).l2(1e-4).list()
				// new BayesTreeActivationFunction(0, false, -1198)

				.layer(0,
						new CustomLayer.Builder().nIn(numInputs).nOut(Constants.numberOfNeurons)
								.activation(Activation.SIGMOID).build())
				.layer(1,
						new CustomLayer.Builder().nIn(Constants.numberOfNeurons).nOut(Constants.numberOfNeurons)
								.activation(Activation.SIGMOID).build())
				.layer(2,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX).nIn(Constants.numberOfNeurons).nOut(Constants.numClasses).build())
				.backprop(true).pretrain(false).build();

		// run the model
		Constants.model = new MultiLayerNetwork(conf);

		Constants.model.init();
		Constants.model.setListeners(new ScoreIterationListener(5));

		// RandomSub ils.instancesToDataSet(ins);
		System.out.println("start");

		ArrayList<Integer> arr = new ArrayList<>();
		INDArray features = trainingData.getFeatures();
		System.out.println(features.shapeInfoToString());
		System.out.println(features.columns());
		for (int i = 0; i < features.columns() - 1; i++) {
			arr.add(i);
		}
		/**
		 * 
		 * TODO : here we set the number of the attributes that is going to be
		 * chosed for each neurons ( max number) => we need to find some
		 * automatic way
		 * 
		 * 
		 */

		int number_of_features_for_each_neuron = 4;
		HashMap<Integer, Boolean> attInexes = new HashMap<>();
		for (int j = 0; j < Constants.numberOfNeurons; j++) {
			Collections.shuffle(arr);
			int[] temp = new int[number_of_features_for_each_neuron];
			for (int i = 0; i < number_of_features_for_each_neuron; i++) {
				temp[i] = arr.get(i);
				attInexes.put(arr.get(i), true);
			}

			Constants.attributesIndexes.put(j, temp);

		}

		// class configuration for each neuron

		ArrayList<Integer> tmp1 = new ArrayList<Integer>();

		for (int c = 0; c < Constants.numClasses - 1; c++) {
			// for 4 classes -> it is set only for mnist dataset ( to be changed
			// )
			for (int i = 0; i < (int) (Constants.numberOfNeurons / Constants.numClasses); i++) {
				tmp1.add(c);
			}
		}

		while (tmp1.size() < Constants.numberOfNeurons)
			tmp1.add(Constants.numClasses - 1);

		for (int l = 0; l < Constants.numberOfLayers; l++) {

			@SuppressWarnings("unchecked")
			ArrayList<Integer> tmp2 = (ArrayList<Integer>) tmp1.clone();
			Collections.shuffle(tmp2);
			Constants.classChosedArray.put(l, tmp2);
		}

		Instances trainSet2 = null;
		trainSet2 = _utils.dataset2Instances(trainingData);

		NumericToNominal convert = new NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "" + (trainSet2.numAttributes()); // range of variables to
		convert.setOptions(options);
		convert.setInputFormat(trainSet2);
		trainSet2 = weka.filters.Filter.useFilter(trainSet2, convert);
		trainSet2.setClassIndex(trainSet2.numAttributes() - 1);
		DataSet tempTrainSet = _utils.instancesToDataSet(trainSet2);

		
		double trainSize = trainSet2.size();
		Constants.numBatches =(int) ( trainSize / (double)Constants.batchSize);
		

		for (int i = 0; i < 150; i++) {
			// in the first iteration do the bagging test and the each batch
			// test :D
			if (i == 0) {
				// DataSet testSet = mnistTest.next();
				// mnistTest.reset();
				// Instances testInstances = _utils.dataset2Instances(testSet);
				// double batchTest = batch_test(batchNum, batchSize,
				// tempTrainSet, trainSet2, testSet,testInstances);
				// double baggingTest = bagging_test(batchNum, batchSize,
				// tempTrainSet, trainSet2, testSet,testInstances);
				// String path =
				// "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/"
				// +
				// "phase_3/without_depth_limit/without_normalization/batch_&_bagging_results.txt";
				// File file = new File(path);
				// BufferedWriter out = new BufferedWriter(new
				// FileWriter(file));
				// out.write("number of batches :\t" + batchNum+"\n");
				// out.write("size of batches :\t" + batchSize+"\n");
				// out.write("size of testSet :\t" +
				// testSet.numExamples()+"\n");
				// out.write("avg batch result:\t" + batchTest+"\n ");
				// out.write("bagging result:\t" + baggingTest);
				// out.close();

			}
			for (int b = 0; b < Constants.numBatches; b++) {

				DataSet set = getBatchTrainSet(b, Constants.batchSize, tempTrainSet, trainSet2);
				Constants.model.fit(set);
			}

			if (i % 2 == 0) {
				Constants.isEvaluating = true;
				log.info("Evaluate model....");

				Evaluation eval = new Evaluation(Constants.numClasses); // create an

					System.out.println(Constants.isEvaluating);
					_utils.setLabels(testData.getLabels(), Constants.isEvaluating, false);
					INDArray output = Constants.model.output(testData.getFeatures());

					eval.eval(testData.getLabels(), output);



				String path = "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/phase_3/without_depth_limit/without_normalization/3/resultIteration_"
						+ i;
				File file = new File(path);
				BufferedWriter out = new BufferedWriter(new FileWriter(file));
				out.write(eval.stats() + "\n" + Constants.model.score());
				// System.out.println(eval.stats());

				out.close();
				Constants.isEvaluating = false;

			}

		}

		Constants.isEvaluating = true;
		Evaluation eval = new Evaluation(Constants.numClasses); // create an evaluation
			_utils.setLabels(testData.getLabels(), Constants.isEvaluating, false);
			INDArray output = Constants.model.output(testData.getFeatures());
			eval.eval(testData.getLabels(), output); // check the prediction
		
		log.info(eval.stats());

	}

	private static DataSet getBatchTrainSet(int batchNumber, int batchRate, DataSet trainSet, Instances training) {

		INDArray features = trainSet.getFeatures();
		INDArray labels = trainSet.getLabels();
		int start = batchNumber * batchRate;
		int end = (batchNumber + 1) * batchRate;

		INDArray batchTrain_features = features.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
		INDArray batchTrain_labels = labels.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());

		DataSet set = new DataSet(batchTrain_features, batchTrain_labels);
		List<Instance> list = training.subList(start, end);
		double[] labels_list = new double[list.size()];
		for (int i = 0; i < list.size(); i++)
			labels_list[i] = list.get(i).classValue();
		// System.out.println(labels.length());
		// System.out.println(batchTrain_labels.shapeInfoToString());
		// System.out.println(batchTrain_labels);
		// System.out.println(labels_list[labels_list.length - 1]);
		Constants.trainInstancesLabel = Nd4j.create(labels_list).transpose();
		// System.out.println(Constants.trainInstancesLabel.shapeInfoToString());
		// System.out.println(set.numExamples());

		return set;

	}

	// each tree accuracy on =0h and bagging of trees
	private static double batch_test(int batchNum, int batchSize, DataSet trainSet, Instances training, DataSet testSet,
			Instances testInstances) throws Exception {

		double avgAccuracy = 0d;

		for (int b = 0; b < batchNum; b++) {
			DataSet set = getBatchTrainSet(b, batchSize, trainSet, training);
			Instances train = _utils.dataset2Instances(set);
			HoeffdingTree hf = new HoeffdingTree();
			hf.buildClassifier(train);
			weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(train);
			eval.evaluateModel(hf, testInstances);
			avgAccuracy += eval.pctCorrect();
		}

		return avgAccuracy / batchNum;

	}

	private static double bagging_test(int batchNum, int batchSize, DataSet trainSet, Instances training,
			DataSet testSet, Instances testInstances) throws Exception {
		double avgAccuracy = 0d;
		ArrayList<HoeffdingTree> trees;

		for (int b = 0; b < batchNum; b++) {

			DataSet trainDataset = getBatchTrainSet(b, batchSize, trainSet, training);
			trees = new ArrayList<>();
			for (int i = 0; i < Constants.numberOfNeurons; i++) {
				INDArray bag = _utils.getSubDataset(Constants.attributesIndexes.get(i), trainDataset);

				Instances train = _utils.ndArrayToInstances(bag);
				HoeffdingTree hf = new HoeffdingTree();
				hf.buildClassifier(train);
				trees.add(hf);
			}

			double correct = 0d;
			int[][] classPredicted = new int[testSet.numExamples()][Constants.numClasses];
			for (int j = 0; j < trees.size(); j++) {
				INDArray temp = _utils.getSubDataset(Constants.attributesIndexes.get(j), testSet);
				Instances test = _utils.ndArrayToInstances(temp);
				Iterator<Instance> it = test.iterator();
				int counter = 0;
				INDArray bag = _utils.getSubDataset(Constants.attributesIndexes.get(j), trainDataset);
				Instances train = _utils.ndArrayToInstances(bag);
				weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(train);
				eval.evaluateModel(trees.get(j), test);
				// System.out.println(eval.pctCorrect());
				while (it.hasNext()) {

					Instance inst = it.next();
					// for (int i = 0; i < trees.size(); i++) {
					// if (i == 26) {
					// System.out.println("i == 26");
					// }
					// if ((int) trees.get(i).classifyInstance(inst) == 26)
					// System.out.println("(int)
					// trees.get(i).classifyInstance(inst) == 26");
					// System.out.println(trees.get(j).classifyInstance(inst));
					classPredicted[counter][(int) trees.get(j).classifyInstance(inst)]++;
					// }
					counter++;
				}

			}

			for (int i = 0; i < classPredicted.length; i++) {
				int max = Integer.MIN_VALUE;
				int max_indx = -1;
				for (int j = 0; j < Constants.numClasses; j++) {
					if (classPredicted[i][j] > max) {
						max = classPredicted[i][j];
						max_indx = j;
					}

				}

				if (max_indx == (int) testInstances.get(i).classValue())
					correct++;

			}

			avgAccuracy += correct / testSet.numExamples();

		}

		return avgAccuracy / batchNum;
	}

	@SuppressWarnings("unused")
	private static Instances createProperDataset(INDArray in, boolean training) {
		Instances instances = null;

		try {
			INDArray label = null;

			if (training == false)
				label = Constants.testInstancesLabel;
			else
				label = Constants.trainInstancesLabel;

			INDArray dataset = Nd4j.concat(1, in, label);

			instances = _utils.ndArrayToInstances(dataset);
			instances.setClassIndex(instances.numAttributes() - 1);

			if (!instances.classAttribute().isNominal()) {
				NumericToNominal convert = new NumericToNominal();
				String[] options = new String[2];
				options[0] = "-R";
				options[1] = "" + (instances.classIndex() + 1); // range of
				// variables to
				// make numeric
				convert.setOptions(options);
				convert.setInputFormat(instances);
				instances = weka.filters.Filter.useFilter(instances, convert);
			}
		} catch (WekaException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return instances;

	}
}