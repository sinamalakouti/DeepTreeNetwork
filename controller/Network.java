package controller;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import neuralnetwork.ActivationFunction;
import neuralnetwork.CustomLayer;
import tree.DecisionTree;
import utils._utils;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.NumericToNominal;


public class Network {

    private static Logger log = LoggerFactory.getLogger(Network.class);

	public static void main(String[] args) throws Exception {

		// First: get the dataset using the record reader. CSVRecordReader handles
		// loading/parsing
		int numLinesToSkip = 0;
		char delimiter = ',';
		RecordReader recordReader = new CSVRecordReader();
		recordReader.initialize(new FileSplit(  new File("iris.txt")   ));

		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("iris.txt"));
		String[] options = new String[1];
		options[0] = "-H";
		loader.setOptions(options);
		Instances dataset = loader.getDataSet();
		
		NumericToNominal convert = new NumericToNominal();
		options = new String[2];
		options[0] = "-R";
		options[1] = "5"; // range of variables to make numeric
		convert.setOptions(options);
		convert.setInputFormat(dataset);
		dataset = weka.filters.Filter.useFilter(dataset, convert);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		// Second: the RecordReaderDataSetIterator handles conversion to DataSet
		// objects, ready for use in neural network
		int labelIndex = 4; // 5 values in each row of the iris.txt CSV: 4 input features followed by an
		// integer label (class) index. Labels are the 5th value (index 4) in each row
		int numClasses = 3; // 3 classes (types of iris flowers) in the iris data set. Classes have integer
		// values 0, 1 or 2
		int batchSize = 150; // Iris data set: 150 examples total. We are loading all of them into one
		// DataSet (not recommended for large data sets)

		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
		DataSet allData = iterator.next();
		allData.shuffle();
		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65); // Use 65% of data for training

		DataSet trainingData = testAndTrain.getTrain();
		DataSet testData = testAndTrain.getTest();

		java.util.Random rand = new java.util.Random(3);
		Instances randData = new Instances(dataset);
		randData.setClassIndex(randData.numAttributes() - 1);
		randData.randomize(rand);
		randData.stratify(10);

		Instances training = randData.trainCV(10, 0);
		training.setClassIndex(training.numAttributes() - 1);
		Instances test = randData.testCV(10, 0);
		test.setClassIndex(test.numAttributes() - 1);
		training.setClassIndex(training.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		trainingData = _utils.instancesToDataSet(training);
		testData = _utils.instancesToDataSet(test);

		// We need to normalize our data. We'll use NormalizeStandardize (which gives us
		// mean 0, unit variance):
//		DataNormalization normalizer = new NormalizerStandardize();
//		normalizer.fit(trainingData); // Collect the statistics (mean/stdev) from the training data. This does not
//		// modify the input data
//		normalizer.transform(trainingData); // Apply normalization to the training data
//		normalizer.transform(testData); // Apply normalization to the test data. This is using statistics calculated
//		// from the *training* set

		final int numInputs = 4;

		int outputNum = 3;
		long seed = 6;
		log.info("Build model....");
		
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.NONE)	
				.weightInit(WeightInit.XAVIER).updater(new Sgd(0.1)).l2(1e-4).list()
				
			
//				.layer(0,
//						new DenseLayer.Builder().nIn(4).nOut(3)
//						.activation(new ActivationFunction(training, test, false))
				.layer(0,
						new DenseLayer.Builder().nIn(4).nOut(2)
						.activation(Activation.SIGMOID)
//						.activation(new ActivationFunction(training, test, false))
//				.layer(0,
//						new ElementWiseMultiplicationLayer.Builder().nIn(4).nOut(4)
//						.activation(new ActivationFunction(training, test, false))

						.build())
				.layer(1, new CustomLayer.Builder().nIn(2).nOut(3)
						.activation(new ActivationFunction(training, test, false))
						.build())
				//         todo : we should obviously change this one

				.layer(2,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
				.backprop(false).pretrain(false).build();

		// run the model
		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		model.init();
		model.setListeners(new ScoreIterationListener(100));
		DecisionTree tree = new DecisionTree(0, test);
	
	
		for (int i = 0; i < 1; i++) {
			model.fit(trainingData);
		}
		
		weka.classifiers.Evaluation eval1 = new weka.classifiers.Evaluation(test);
		eval1.evaluateModel(tree, test);
		
		// evaluate the model on the test set
		Evaluation eval = new Evaluation(3);
		INDArray output = model.output(testData.getFeatures());
		eval.eval(testData.getLabels(), output);
		log.info(eval.stats());
	}
}