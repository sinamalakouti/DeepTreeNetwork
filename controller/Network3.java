package controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Network3 {

	private static Logger log = LoggerFactory.getLogger(Network3.class);

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

	public static void main(String[] args) throws Exception {

		// TODO Nd4j.setDataType(Type.DOUBLE);

		String[] options;

		NumericToNominal convert = new NumericToNominal();
		int batchNum = 600;
		// Constants.maximumDepth = 5;
		// for weights normalization
		Constants.weightLayerMin = new double[2];
		Constants.weightLayerMin[0] = Double.POSITIVE_INFINITY;
		Constants.weightLayerMin[1] = Double.POSITIVE_INFINITY;
		Constants.weightLayerMax = new double[2];
		Constants.weightLayerMax[0] = Double.NEGATIVE_INFINITY;
		Constants.weightLayerMax[1] = Double.NEGATIVE_INFINITY;

		final int numInputs = 784;
		int outputNum = 10;
		Constants.numberOfLayers = 2;
		Constants.numberOfNeurons = 40;
		Constants.batchSize = 100;
		double numberTrainExamples = 60000d;
		Constants.numBatches = (int) ( (numberTrainExamples) / Constants.batchSize); 

		
		log.info("Build model....");
		// org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(6)

				.trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
				.weightInit(WeightInit.XAVIER).updater(new Sgd(0.1)).l2(1e-4).list()
				// new BayesTreeActivationFunction(0, false, -1198)

				.layer(0, new CustomLayer.Builder().nIn(numInputs).nOut(40).activation(Activation.SIGMOID).build())
				.layer(1, new CustomLayer.Builder().nIn(40).nOut(40).activation(Activation.SIGMOID).build())
				.layer(2,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX).nIn(40).nOut(outputNum).build())
				.backprop(true).pretrain(false).build();

		// run the model
		Constants.model = new MultiLayerNetwork(conf);

		Constants.model.init();
		Constants.model.setListeners(new ScoreIterationListener(5));

//		Constants.test = test;
//		Constants.train = training;

		// RandomSub ils.instancesToDataSet(ins);
		System.out.println("start");
//		System.out.println(trainingData.numExamples());

		ArrayList<Integer> arr = new ArrayList<>();
		for (int i = 0; i < 784; i++) {
			arr.add(i);
		}
		int max = 784 / 30;
		HashMap<Integer, Boolean> attInexes = new HashMap<>();
		for (int j = 0; j < 40; j++) {
			Collections.shuffle(arr);
			int[] temp = new int[max];
			for (int i = 0; i < max; i++) {
				temp[i] = arr.get(i);
				attInexes.put(arr.get(i), true);
			}

			Constants.attributesIndexes.put(j, temp);

		}

		Constants.numClasses= 10;
		Constants.numberOfLayers =2;
		Constants.numberOfNeurons =40;
		// class configuration for each neuron
		ArrayList<Integer> tmp1 = new ArrayList<Integer>();
//
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


		// setupe the project :
		int batchSize = 100;
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 6);
		DataSetIterator mnistTest = new MnistDataSetIterator(1000, false, 6);

		Instances trainSet2 = null, trainTemp = null;
		int c = 0;

		while (mnistTrain.hasNext()) {
			DataSet set = mnistTrain.next();
			if (c == 0) {
				trainSet2 = _utils.dataset2Instances(set);
			} else {
				trainTemp = _utils.dataset2Instances(set);
				for (int i = 0; i < trainTemp.size(); i++)
					trainSet2.add(trainTemp.get(i));
			}

			c++;
		}
		mnistTrain.reset();

		convert = new NumericToNominal();
		options = new String[2];
		options[0] = "-R";
		options[1] = "" + (trainSet2.numAttributes()); // range of variables to
		convert.setOptions(options);
		convert.setInputFormat(trainSet2);
		trainSet2 = weka.filters.Filter.useFilter(trainSet2, convert);
		trainSet2.setClassIndex(trainSet2.numAttributes() - 1);
		DataSet tempTrainSet = _utils.instancesToDataSet(trainSet2);

		batchSize = 100;
		int trainSize = trainSet2.size();
		batchNum = trainSize / batchSize;
		System.out.println(trainSet2.size());
		System.out.println(tempTrainSet.numExamples());
		System.out.println(batchNum);
		System.out.println(batchSize);
		// System.out.println(mnistTest.getLabels().size());

		for (int i = 0; i < 150; i++) {
			// in the first iteration do the bagging test and the each batch
			// test :D
			for (int b = 0; b < 600; b++) {

				DataSet set = getBatchTrainSet(b, 100	, tempTrainSet, trainSet2);
//				if (i == 0) {
//
//					Instances batchTrainInstance = _utils.dataset2Instances(set);
//
//					// batch test :
//
//					if (batchTree == null) {
//						batchTree = new HoeffdingTree();
//						batchTree.buildClassifier(batchTrainInstance);
//					} else {
//						Iterator<Instance> it = batchTrainInstance.iterator();
//						while (it.hasNext())
//							batchTree.updateClassifier(it.next());
//					}
//
//					// bagging test:
//					for (int t = 0; t < Constants.numberOfNeurons; t++) {
//						INDArray bag = _utils.getSubDataset(Constants.attributesIndexes.get(t), set);
//						Instances bagInstances = _utils.ndArrayToInstances(bag);
//						if (baggingTrees[t] == null) {
//							baggingTrees[t] = new HoeffdingTree();
//							baggingTrees[t].buildClassifier(bagInstances);
//						} else {
//							Iterator<Instance> it = bagInstances.iterator();
//							while (it.hasNext()) {
//								baggingTrees[t].updateClassifier(it.next());
//							}
//						}
//
//					}
//
//				}

				Constants.model.fit(set);
			}

//			if (i == 0) {
//
//				DataSet testSet = mnistTest.next();
//				mnistTest.reset();
//
////				String path = "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/"
////						+ "phase_3/without_depth_limit/without_normalization/batch_&_bagging_results.txt";
////				File file = new File(path);
////				BufferedWriter out = new BufferedWriter(new FileWriter(file));
////				out.write("number of batches :\t" + Constants.numBatches + "\n");
////				out.write("size of batches :\t" + Constants.batchSize + "\n");
////				out.write("size of testSet :\t" + testSet.numExamples() + "\n");
//				int batchCounter = 0;
//				int baggingCounter = 0;
//				int[][] classPredicted = new int[testSet.numExamples()][Constants.numClasses];
//				Instances testInstances = null;
//				for (int t = 0; t < baggingTrees.length; t++) {
//					INDArray tempTest = _utils.getSubDataset(Constants.attributesIndexes.get(t), testSet);
//					testInstances = _utils.ndArrayToInstances(tempTest);
//					Iterator<Instance> testIterator = testInstances.iterator();
//					int s = 0;
//					while( testIterator.hasNext()){
//						Instance sample = testIterator.next();
//						classPredicted[s][(int) baggingTrees[t].classifyInstance(sample)]++;
//						s++;
//					}
//					
//					
//				}
//				
//				
//
//				for (int s = 0; s < classPredicted.length; s++) {
//					
//					double max_prediction = Double.MIN_VALUE;
//					int BaggingClass = -1;
//
//					for( c = 0 ; c < Constants.numClasses ; c++ ){
//						if (classPredicted[s][c] > max_prediction) {
//							max_prediction = classPredicted[s][c];
//							BaggingClass = c;
//						}	
//						
//					}
//					
//					if( BaggingClass == testInstances.get(s).classValue())
//						baggingCounter++;
//					
//				}
//					
//
//				
//			    testInstances = _utils.dataset2Instances(testSet);
//				Iterator<Instance> testIterator = testInstances.iterator();
//				while (testIterator.hasNext()) {
//					Instance sample = testIterator.next();
//					double batchPrediction = batchTree.classifyInstance(sample);
//					
//						
//					
//					
//					if (batchPrediction == sample.classValue())
//						batchCounter++;
////					if (BaggingClass == sample.classValue())
////						baggingCounter++;
//
//				}
//
//				
//				System.out.println((double) batchCounter / testInstances.size());
//				System.out.println((double) batchCounter / (double) testInstances.size());
////				out.write("avg batch result:\t" +(double) batchCounter / testInstances.size() + "\n ");
////				out.write("bagging result:\t" + (double)baggingCounter / testInstances.size());
////				out.close();
//			}

			if (i % 2 == 0) {
				Constants.isEvaluating = true;
				log.info("Evaluate model....");

				Evaluation eval = new Evaluation(outputNum); // create an

				while (mnistTest.hasNext()) {

					DataSet next = mnistTest.next();
					System.out.println(Constants.isEvaluating);
					_utils.setLabels(next.getLabels(), Constants.isEvaluating, false);
					INDArray output = Constants.model.output(next.getFeatures());

					eval.eval(next.getLabels(), output);
				}
				mnistTest.reset();

////				 String path =
////				 "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/"
////						+ "phase_3/without_depth_limit/without_normalization/resultIteration_"+ i;
////				 File file = new File(path);
////				 BufferedWriter out = new BufferedWriter(new
////				 FileWriter(file));
//				 out.write(eval.stats() + "\n" + Constants.model.score());
				System.out.println(eval.stats() + "\n" + Constants.model.score());
				

//				 out.close();
				Constants.isEvaluating = false;

			}

		}

		Constants.isEvaluating = true;
		Evaluation eval = new Evaluation(outputNum); // create an evaluation
		// object with 10
		// possible classes
		int counter = 0;
		while (mnistTest.hasNext()) {

			DataSet next = mnistTest.next();
			_utils.setLabels(next.getLabels(), Constants.isEvaluating, false);
			INDArray output = Constants.model.output(next.getFeatures());
			eval.eval(next.getLabels(), output); // check the prediction
			// against the true
			// class
			counter++;
		}
		System.out.println(counter);
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
		Constants.trainInstancesLabel = Nd4j.create(labels_list).transpose();
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
					// 	.out.println("i == 26");
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
}