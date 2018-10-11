package controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer.Type;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ND4JTestUtils;
import org.nd4j.linalg.util.NDArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import neuralnetwork.BayesTreeActivationFunction;
import neuralnetwork.CustomLayer;
import scala.collection.immutable.Stream.Cons;
import utils.Constants;
import utils._utils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Network {

	private static Logger log = LoggerFactory.getLogger(Network.class);

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
		Constants.maximumDepth = 3;
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
		DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 6);

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

		for (int i = 0; i < 150; i++) {
			// for ( int b = 0; b < batchNum ; b ++) {

			// DataSet set = getBatchTrainSet(b, batchSize, trainingData,
			// training);
			//
			// int counter = 0 ;
			while (mnistTrain.hasNext()) {
				DataSet set = mnistTrain.next();
				Constants.model.fit(set);
				// counter ++;
			}
			mnistTrain.reset();

			// }
			// model.fit(trainingData);

			if (i % 2 == 0) {
				Constants.isEvaluating = true;
				// Evaluation eval = new Evaluation(outputNum);
				// System.out.println(mnistTest.());
				// INDArray output =
				// Constants.model.output(testData.getFeatures());
				System.out.println("sdaf");
				// eval.eval(testData.getLabels(), output);

				log.info("Evaluate model....");
				// counter = 0;
				Evaluation eval = new Evaluation(outputNum); // create an
																// evaluation
																// object with
																// 10 possible
																// classes
				while (mnistTest.hasNext()) {

					DataSet next = mnistTest.next();
					System.out.println(Constants.isEvaluating);
					_utils.setLabels(next.getLabels(), Constants.isEvaluating, false);
					INDArray output = Constants.model.output(next.getFeatures());

					eval.eval(next.getLabels(), output);
				}
				mnistTest.reset();



				String path = "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/new/without_normalization/resultIteration_"
						+ i;
				File file = new File(path);
				BufferedWriter out = new BufferedWriter(new FileWriter(file));
				out.write(eval.stats() + "\n" + Constants.model.score());
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
}