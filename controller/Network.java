package controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.jfree.data.general.DatasetUtilities;
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

import ch.qos.logback.core.net.SyslogOutputStream;
import neuralnetwork.BayesTreeActivationFunction;
import neuralnetwork.CustomLayer;
import neuralnetwork.LossBayesTree;
import neuralnetwork.OutputCustomLayer;
import scala.collection.immutable.Stream.Cons;
import sigmoid.MySigmoidActivationFunction;
import utils.Constants;
import utils._utils;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Network {

	private static Logger log = LoggerFactory.getLogger(Network.class);

	public static void main(String[] args) throws Exception {

		// First: get the dataset using the record reader. CSVRecordReader handles
		// loading/parsing

 
//  <iris datast>		
		
//		CSVRecordReader recordReader = new CSVRecordReader();
//		recordReader.initialize(new FileSplit(new File("iris.txt")));
//
//		CSVLoader loader = new CSVLoader();
//		loader.setSource(new File("iris.txt"));
//		String[] options = new String[1];
//		options[0] =  "-H";
//		loader.setOptions(options);
//        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
//        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
//        int batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
//
//		Instances dataset = loader.getDataSet();
//		NumericToNominal convert = new NumericToNominal();
//		options = new String[2];
//		options[0] = "-R";
//		options[1] = "" + (dataset.numAttributes()); // range of variables to make numeric 
//		convert.setOptions(options);
//		convert.setInputFormat(dataset);
//		dataset = weka.filters.Filter.useFilter(dataset, convert);
//		dataset.setClassIndex(dataset.numAttributes() - 1);
//	
//		// DataSet (not recommended for large data sets)
//
//		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
//		DataSet allData = iterator.next();
//		allData.shuffle();
//		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65); // Use 65% of data for training
//
//		DataSet trainingData = testAndTrain.getTrain();
//		DataSet testData = testAndTrain.getTest();
//
//		java.util.Random rand = new java.util.Random(3);
//		Instances randData = new Instances(dataset);
//		randData.setClassIndex(randData.numAttributes() - 1);
//		randData.randomize(rand); 
//		randData.stratify(10);
//
//		Instances training = randData.trainCV(10, 0);
//		training.setClassIndex(training.numAttributes() - 1);
//		Instances test = randData.testCV(10, 0);
//		test.setClassIndex(test.numAttributes() - 1);
//		training.setClassIndex(training.numAttributes() - 1);
//		test.setClassIndex(test.numAttributes() - 1);
//		trainingData = _utils.instancesToDataSet(training);
//		testData = _utils.instancesToDataSet(test);
		
	
//		</irisdataset>
		
//		<isolet5>
		
//		
//		CSVRecordReader recordReader = new CSVRecordReader();
//		recordReader.initialize(new FileSplit(new File("isolet5.csv")));
//
//		CSVLoader loader = new CSVLoader();
//		loader.setSource(new File("isolet5.csv"));
//		String[] options = new String[1];
//		options[0] =  "-H";
//		loader.setOptions(options);
//        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
//        int numClasses = 26;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
//        int batchSize = 500;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
//
//		Instances dataset = loader.getDataSet();
//		NumericToNominal convert = new NumericToNominal();
//		options = new String[2];
//		options[0] = "-R";
//		options[1] = "" + (dataset.numAttributes()); // range of variables to make numeric 
//		convert.setOptions(options);
//		convert.setInputFormat(dataset);
//		dataset = weka.filters.Filter.useFilter(dataset, convert);
//		dataset.setClassIndex(dataset.numAttributes() - 1);
//	
//		// DataSet (not recommended for large data sets)
//
//		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
//		DataSet allData = iterator.next();
//		allData.shuffle();
//		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65); // Use 65% of data for training
//
//		DataSet trainingData = testAndTrain.getTrain();
//		DataSet testData = testAndTrain.getTest();
//
//		java.util.Random rand = new java.util.Random(3);
//		Instances randData = new Instances(dataset);
//		randData.setClassIndex(randData.numAttributes() - 1);
//		randData.randomize(rand); 
//		randData.stratify(10);
//
//		Instances training = randData.trainCV(10, 0);
//		training.setClassIndex(training.numAttributes() - 1);
//		Instances test = randData.testCV(10, 0);
//		test.setClassIndex(test.numAttributes() - 1);
//		training.setClassIndex(training.numAttributes() - 1);
//		test.setClassIndex(test.numAttributes() - 1);
//		trainingData = _utils.instancesToDataSet(training);
//		testData = _utils.instancesToDataSet(test);
		
		
		
		
		
		
//		</isolet5>
//		<mnistdataset>
		
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
		

		// Second: the RecordReaderDataSetIterator handles conversion to DataSet
//		// objects, ready for use in neural network
		int labelIndex = 4; // 5 values in each row of the iris.txt CSV: 4 input features followed by an
		// integer label (class) index. Labels are the 5th value (index 4) in each row
		int batchSize = 100; // Iris data set: 150 examples total. We are loading all of them into one
		int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
		int testSize = dataset.numInstances() - trainSize;
		Instances training = new Instances(dataset, 0, trainSize);
		Instances test = new Instances(dataset, trainSize, testSize);
		training.randomize(new java.util.Random(0));
//
		DataSet trainingData = _utils.instancesToDataSet(training);
		DataSet testData = _utils.instancesToDataSet(test);
		int  batchNum = trainingData.numExamples() / batchSize;
		Constants.maximumDepth =3;
//		</mnistdataset>
		
		

//		<amazondataset>
		
		
//		DataSource source = new DataSource("amazon.arff");
//		Instances dataset = source.getDataSet();
//		System.out.println(dataset.size());
//		String[] options;
//		System.out.println(dataset.attribute(999).isNumeric());
//		System.out.println(dataset.numAttributes());
//		NumericToNominal convert = new NumericToNominal();
//		options = new String[2];
//		options[0] = "-R";
//		System.out.println(dataset.attribute(999).isNumeric());
//		options[1] = "" + (dataset.numAttributes()); // range of variables to make numeric 
//		convert.setOptions(options);
//		convert.setInputFormat(dataset);
//		dataset = weka.filters.Filter.useFilter(dataset, convert);
//		System.out.println(dataset.attribute(999).isNumeric());
//		dataset.setClassIndex(dataset.numAttributes() - 1);
//		
//
//		int labelIndex = 4; // 5 values in each row of the iris.txt CSV: 4 input features followed by an
//		// integer label (class) index. Labels are the 5th value (index 4) in each row
//		int numClasses = 3; // 3 classes (types of iris flowers) in the iris data set. Classes have integer
//		// values 0, 1 or 2
//		int batchSize = 150; // Iris data set: 150 examples total. We are loading all of them into one
//		int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
//		int testSize = dataset.numInstances() - trainSize;
//		Instances training = new Instances(dataset, 0, trainSize);
//		Instances test = new Instances(dataset, trainSize, testSize);
//		training.randomize(new java.util.Random(0));
//
//		DataSet trainingData = _utils.instancesToDataSet(training);
//		DataSet testData = _utils.instancesToDataSet(test);
//		
//		
//		</amazondataset>
		
		
		
		
//		training.setClassIndex(training.numAttributes() -1);

		// We need to normalize our data. We'll use NormalizeStandardize (which gives us
		// mean 0, unit variance):
//		DataNormalization normalizer = new NormalizerStandardize();
//		normalizer.fit(trainingData); // Collect the statistics (mean/stdev) from the training data. This does not
//		// modify the input data
//		normalizer.transform(trainingData); // Apply normalization to the training data
//		normalizer.transform(testData); // Apply normalization to the test data. This is using statistics calculated
//		// from the *training* set

		final int numInputs = 784;

		int outputNum = 10;
		long seed = 6;
		log.info("Build model....");
//org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)

				.trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
				.weightInit(WeightInit.XAVIER).updater(new Sgd(0.1)).l2(1e-4).list()


				.layer(0, new CustomLayer.Builder().nIn(numInputs).nOut(40).activation(new BayesTreeActivationFunction(0, false,-1))
						.build())
				.layer(1,
						new CustomLayer.Builder().nIn(40).nOut(40).activation(new BayesTreeActivationFunction(0, false,-2)).build())
				.layer(2,
						new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
								.activation(Activation.SOFTMAX).nIn(40).nOut(outputNum).build())
				.backprop(true).pretrain(false).build();

		// run the model
		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		model.init();
		model.setListeners(new ScoreIterationListener(5));

		Constants.test = test;
		Constants.train = training;

//		RandomSub ils.instancesToDataSet(ins);
		System.out.println("start");
		System.out.println(trainingData.numExamples());

		ArrayList<Integer> arr = new ArrayList<>();
		
		for (int i = 0; i < training.numAttributes() - 1; i++) {
			arr.add(i);
		}
		int max =  784 / 30;
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

		if ( attInexes.containsValue(false)) {
			System.out.println("ajab badbakhti gir kardim");
			System.exit(0);
		}
		
		
//		class configuration for each neuron
		ArrayList<Integer> tmp1= new ArrayList<Integer>(40);
		ArrayList<Integer> tmp2= new ArrayList<Integer>(40);

		for (int i = 0 ; i < 10 ; i++)
		{
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
		
		for (int i = 0; i < 200; i++) {
			for ( int b = 0;  b < batchNum ; b ++) {
				
				
				DataSet set = getBatchTrainSet(b, batchSize, trainingData, training);
				
				
				model.fit(set	);
//				ModelSerializer.writeModel(model, "laylaylay", true);
				
				

			}
			
			
			
			if ( i % 5 == 0) {
				Constants.isEvaluating = true;
				Evaluation eval = new Evaluation(outputNum);
				INDArray output = model.output(testData.getFeatures());
				System.out.println("sdaf");
				eval.eval(testData.getLabels(), output);
				
				 String path = "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/second/Incremental_and_classConfiguration/resultIteration_"+ i;
				  File file = new File (path);
				  BufferedWriter out = new BufferedWriter(new FileWriter(file)); 
				  out.write(eval.stats());
				  out.close();
				  Constants.isEvaluating = false;


			}


		}


		Evaluation eval = new Evaluation(outputNum);
		INDArray output = model.output(testData.getFeatures());
		eval.eval(testData.getLabels(), output);
		log.info(eval.stats());
		
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