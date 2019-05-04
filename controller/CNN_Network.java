package controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.List;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import neuralnetwork.CustomLayer;
import utils.Constants;
import utils._utils;

import weka.core.Instance;
import weka.core.Instances;

import weka.filters.unsupervised.attribute.NumericToNominal;

public class CNN_Network {

	private static Logger log = LoggerFactory.getLogger(CNN_Network.class);

	@SuppressWarnings("unused")
	public static void main(String[] args) throws Exception {

		// TODO Nd4j.setDataType(Type.DOUBLE);

		String[] options;

		NumericToNominal convert = new NumericToNominal();

		Constants.weightLayerMin = new double[2];
		Constants.weightLayerMin[0] = Double.POSITIVE_INFINITY;
		Constants.weightLayerMin[1] = Double.POSITIVE_INFINITY;
		Constants.weightLayerMax = new double[2];
		Constants.weightLayerMax[0] = Double.NEGATIVE_INFINITY;
		Constants.weightLayerMax[1] = Double.NEGATIVE_INFINITY;

		final int numInputs = 784;
		int outputNum = 10;
		log.info("Build model....");
		Constants.batchSize = 100;
		double numberTrainExamples = 60000d;
		Constants.numBatches = (int) ((numberTrainExamples) / Constants.batchSize);
		Constants.numClasses = 10;
		int feature_ratio = 10;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(6)

				.weightInit(WeightInit.XAVIER).updater(new Sgd(0.1)).l2(1e-4).list()

				.layer(0,
						new CustomLayer.Builder().nIn(numInputs).nOut(Constants.numberOfNeurons)
								.activation(Activation.SIGMOID).build())
				.layer(1,
						new CustomLayer.Builder().nIn(Constants.numberOfNeurons).nOut(Constants.numberOfNeurons)
								.activation(Activation.SIGMOID).build())
				.layer(2,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.activation(Activation.SOFTMAX).nIn(Constants.numberOfNeurons).nOut(outputNum).build())
				.backprop(true).pretrain(false).build();

		// run the model
		Constants.model = new MultiLayerNetwork(conf);

		Constants.model.init();
		Constants.model.setListeners(new ScoreIterationListener(5));
		System.out.println("start");

		// set-up the project :

		DataSetIterator mnistTrain = new MnistDataSetIterator(Constants.batchSize, true, 6);
		DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 6);

		// normalize data set
		DataNormalization scaler = new NormalizerStandardize();
		scaler.fit(mnistTrain);
		mnistTrain.setPreProcessor(scaler);
		mnistTest.setPreProcessor(scaler); // same normalization for better
											// results
		mnistTrain.reset();
		int counter = 0;
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

		// System.out.println(mnistTest.getLabels().size());
		// HoeffdingTree batchTree = null;
		// HoeffdingTree[] baggingTrees = new
		// HoeffdingTree[Constants.numberOfNeurons];

		for (int i = 0; i < 1000; i++) {
			// in the first iteration do the bagging test and the each batch
			// test :D
			for (int b = 0; b < Constants.numBatches; b++) {

				DataSet set = getBatchTrainSet(b, Constants.batchSize, tempTrainSet, trainSet2);
				Constants.model.fit(set);
			}
			if (i % 2 == 0) {
				Constants.isEvaluating = true;
				log.info("Evaluate model....");
				//
				Evaluation eval = new Evaluation(outputNum); // create an
				//
				while (mnistTest.hasNext()) {

					DataSet next = mnistTest.next();
					System.out.println(Constants.isEvaluating);
					_utils.setLabels(next.getLabels(), Constants.isEvaluating, false);
					INDArray output = Constants.model.output(next.getFeatures());

					eval.eval(next.getLabels(), output);
				}
				mnistTest.reset();
				//
				String path = "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/phase4/randomClassConfig/17/resultIteration_"
						+ i;
				// String path =
				////// "resultIteration_"+ i;
				File file = new File(path);
				BufferedWriter out = new BufferedWriter(new FileWriter(file));
				String avglayersTreesDepth = "";
				for (int l = 0; l < Constants.numberOfLayers; l++)
					avglayersTreesDepth = avglayersTreesDepth + " " + Constants.avgHFDepth[l];
				out.write(eval.stats() + "\nerrors\t" + Constants.model.score() + "\n" + avglayersTreesDepth);
				//
				System.out.println(
						eval.stats() + "\n" + "errors:  " + Constants.model.score() + "\n" + avglayersTreesDepth);

				//
				out.close();
				Constants.isEvaluating = false;
				//
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
		// System.out.println(counter);
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

		features.cleanup();
		labels.cleanup();
		batchTrain_features.cleanup();
		batchTrain_labels.cleanup();

		return set;

	}

}