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

public class Network2 {

	private static Logger log = LoggerFactory.getLogger(Network2.class);

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

		// <mnistdataset>

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
		options[1] = "" + (dataset.numAttributes()); // range of variables to
		// make numeric
		convert.setOptions(options);
		convert.setInputFormat(dataset);
		dataset = weka.filters.Filter.useFilter(dataset, convert);
		System.out.println(dataset.attribute(783).isNumeric());
		dataset.setClassIndex(dataset.numAttributes() - 1);

		int batchSize = 100; // Iris data set: 150 examples total. We are
		// loading all of them into one
		int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
		int testSize = dataset.numInstances() - trainSize;
		dataset.randomize(new java.util.Random()); // randomize instance order
		// before splitting dataset

		Instances training = new Instances(dataset, 0, trainSize);
		Instances test = new Instances(dataset, trainSize, testSize);
		dataset = null;

		DataSet trainingData = _utils.instancesToDataSet(training);
		DataSet testData = _utils.instancesToDataSet(test);
		int batchNum = trainingData.numExamples() / batchSize;
		// Constants.maximumDepth = 5;
		// for weights normalization
		Constants.weightLayerMin = new double[2];
		Constants.weightLayerMin[0] = Double.POSITIVE_INFINITY;
		Constants.weightLayerMin[1] = Double.POSITIVE_INFINITY;
		Constants.weightLayerMax = new double[2];
		Constants.weightLayerMax[0] = Double.NEGATIVE_INFINITY;
		Constants.weightLayerMax[1] = Double.NEGATIVE_INFINITY;
		// </mnistdataset>

		final int numInputs = 784;
		int outputNum = 10;
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

		Constants.test = test;
		Constants.train = training;

		// RandomSub ils.instancesToDataSet(ins);
		System.out.println("start");
		System.out.println(trainingData.numExamples());

		ArrayList<Integer> arr = new ArrayList<>();

		for (int i = 0; i < training.numAttributes() - 1; i++) {
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

		// class configuration for each neuron
		ArrayList<Integer> tmp1 = new ArrayList<Integer>(40);
		ArrayList<Integer> tmp2 = new ArrayList<Integer>(40);

		for (int i = 0; i < 10; i++) {
			tmp1.add(i);
			tmp1.add(i);
			tmp1.add(i);
			tmp1.add(i);

			tmp2.add(i);
			tmp2.add(i);
			tmp2.add(i);
			tmp2.add(i);
		}

		Collections.shuffle(tmp1);
		Collections.shuffle(tmp2);

		Constants.classChosedArray.put(0, tmp1);
		Constants.classChosedArray.put(1, tmp2);
		//
		Constants.testInstancesLabel = NDArrayUtil.toNDArray(_utils.getLabels(test)).transpose();
		Constants.trainInstancesLabel = NDArrayUtil.toNDArray(_utils.getLabels(training)).transpose();

		// setupe the project :

		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 6);
		DataSetIterator mnistTest = new MnistDataSetIterator(1000, false, 6);

		int counter = 0;
		INDArray ar = null;
		// while ( mnistTrain.hasNext()){
		//
		// DataSet set = mnistTrain.next();
		// _utils.setLabels(set.getLabels(), false, true);
		// if ( counter == 0)
		// ar = Nd4j.concat(1, set.getFeatures(),Constants.trainInstancesLabel);
		// else
		// {
		// INDArray tm = Nd4j.concat(1,
		// set.getFeatures(),Constants.trainInstancesLabel.dup());
		// ar =Nd4j.concat(0, ar,tm).dup();
		// }
		//
		// counter++;
		// }

		// training = _utils.ndArrayToInstances(ar);
		// System.out.println(ar.getColumn(784));
		// training.setClassIndex(training.numAttributes()-1);
		// training.randomize(new java.util.Random()); // randomize instance
		// order
		// System.out.println(training.numClasses());
		// trainingData = _utils.instancesToDataSet(training);
		//

		// counter = 0 ;
		// ar = null;
		// while ( mnistTest.hasNext()){
		//
		// DataSet set = mnistTest.next();
		// _utils.setLabels(set.getLabels(), true, false);
		// if ( counter == 0)
		// ar = Nd4j.concat(1, set.getFeatures(),Constants.testInstancesLabel);
		// else
		// {
		// INDArray tm = Nd4j.concat(1,
		// set.getFeatures(),Constants.testInstancesLabel);
		// ar =Nd4j.concat(0, ar,tm);
		// }
		//
		// }
		//
		// training = _utils.ndArrayToInstances(ar);
		// Constants.testInstancesLabel = ar.getColumns(784);
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
		trainSize = trainSet2.size();
		batchNum = trainSize / batchSize;
		System.out.println(trainSet2.size());
		System.out.println(tempTrainSet.numExamples());
		System.out.println(batchNum);
		System.out.println(batchSize);
		// System.out.println(mnistTest.getLabels().size());

		for (int i = 0; i < 150; i++) {
			// in the first iteration do the bagging test and the each batch
			// test :D
			if (i == 0) {
				DataSet testSet = mnistTest.next();
				mnistTest.reset();
				double batchTest = batch_test(batchNum, batchSize, tempTrainSet, trainSet2, testSet);
				double baggingTest = bagging_test(batchNum, batchSize, tempTrainSet, trainSet2, testSet);
				String path = "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/"
						+ "phase_3/without_depth_limit/without_normalization/batch_&_bagging_results.txt" + i;
				File file = new File(path);
				BufferedWriter out = new BufferedWriter(new FileWriter(file));
				out.write("number of batches :\t" + batchNum);
				out.write("size of batches :\t" + batchSize);
				out.write("size of testSet :\t" + testSet.numExamples());
				out.write("avg batch result:\t" + batchTest);
				out.write("bagging result:\t" + baggingTest);
				out.close();

			}
			for (int b = 0; b < batchNum; b++) {

				DataSet set = getBatchTrainSet(b, batchSize, tempTrainSet, trainSet2);
				Constants.model.fit(set);
			}

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

				String path = "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/phase_3/without_depth_limit/without_normalization/resultIteration_"
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
		Evaluation eval = new Evaluation(outputNum); // create an evaluation
		// object with 10
		// possible classes
		counter = 0;
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
		// System.out.println(labels.length());
		// System.out.println(batchTrain_labels.shapeInfoToString());
		// System.out.println(batchTrain_labels);
		// System.out.println(labels_list[labels_list.length - 1]);
		Constants.trainInstancesLabel = Nd4j.create(labels_list).transpose();
		// System.out.println(Constants.trainInstancesLabel.shapeInfoToString());
		// System.out.println(set.numExamples());

		return set;

	}

	// each tree accuracy on batch and bagging of trees
	private static double batch_test(int batchNum, int batchSize, DataSet trainSet, Instances training, DataSet testSet)
			throws Exception {

		// ArrayList<HoeffdingTree> baggingTrees = new ArrayList<>();
		// double [] batchTests = new double[batchNum];
		// double [] bagging_batches = new double[batchNum];
		double avgAccuracy = 0d;
		Instances test = _utils.dataset2Instances(testSet);

		for (int b = 0; b < batchNum; b++) {
			DataSet set = getBatchTrainSet(b, batchSize, trainSet, training);
			Instances train = _utils.dataset2Instances(set);
			HoeffdingTree hf = new HoeffdingTree();
			hf.buildClassifier(train);
			weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(train);
			eval.evaluateModel(hf, test);
			avgAccuracy += eval.pctCorrect();
		}

		return avgAccuracy / batchNum;

	}

	private static double bagging_test(int batchNum, int batchSize, DataSet trainSet, Instances training,
			DataSet testSet) throws Exception {

		Instances test = _utils.dataset2Instances(testSet);
		ArrayList<HoeffdingTree> trees = new ArrayList<>();
		for (int b = 0; b < batchNum; b++) {
			DataSet set = getBatchTrainSet(b, batchSize, trainSet, training);
			INDArray bag = _utils.getSubDataset(Constants.attributesIndexes.get(b), trainSet);

			Instances train = _utils.ndArrayToInstances(bag);
			HoeffdingTree hf = new HoeffdingTree();
			hf.buildClassifier(train);
			weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(train);
			trees.add(hf);

		}
		Iterator<Instance> it = test.iterator();
		double correct = 0;
		while (it.hasNext()) {

			Instance inst = it.next();
			int[] classPredicted = new int[trees.size()];
			for (int i = 0; i < trees.size(); i++) {
				classPredicted[(int) trees.get(i).classifyInstance(inst)]++;
			}

			int max = Integer.MIN_VALUE;
			int max_indx = -1;

			for (int i = 0; i < classPredicted.length; i++) {
				if (classPredicted[i] > max) {
					max = classPredicted[i];
					max_indx = i;
				}

			}
			if (max_indx == (int) inst.classValue())
				correct++;

		}

		return correct / test.size();
	}
}