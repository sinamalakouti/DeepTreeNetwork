package controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.NDArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import neuralnetwork.BayesTreeActivationFunction;
import neuralnetwork.CustomLayer;
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
				options[1] = "" + (instances.classIndex() + 1); // range of variables to make numeric
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
		

		int batchSize = 100; // Iris data set: 150 examples total. We are loading all of them into one
		int trainSize = (int) Math.round(dataset.numInstances() * 0.7);
		int testSize = dataset.numInstances() - trainSize;
		dataset.randomize(new java.util.Random());	// randomize instance order before splitting dataset

		
		
		
		Instances training = new Instances(dataset, 0, trainSize);
		Instances test = new Instances(dataset, trainSize, testSize);

		
//		
//        Normalize filterNorm = new Normalize();
//        filterNorm.setInputFormat(training);
//		training = Filter.useFilter(training, filterNorm);
//		test = Filter.useFilter(test, filterNorm);
		dataset = null;

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
		log.info("Build model....");
//org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(6)

				.trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
				.weightInit(WeightInit.XAVIER).updater(new Sgd(0.1)).l2(1e-4).list()


				.layer(0, new CustomLayer.Builder().nIn(numInputs).nOut(40).activation(new BayesTreeActivationFunction(0, false, -1198))
						.build())
				.layer(1,
						new CustomLayer.Builder().nIn(40).nOut(40).activation(new BayesTreeActivationFunction(1, false, -100) ).build())
//				.layer(1,
//						new DenseLayer.Builder().nIn(40).nOut(40).activation(Activation.SIGMOID).build())
//						new OutputLayer.Builder(new LossBayesTree(null))
//								.activation(new BayesTreeActivationFunction(2, true, -3)).nIn(40).nOut(outputNum).build())
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
		
		
	
		for ( int i = 0; i < 200; i++) {
//			for ( int b = 0;  b < batchNum ; b ++) {
				
//				DataSet set = getBatchTrainSet(b, batchSize, trainingData, training);
//				INDArray z = set.getFeatures().dup();
////				System.out.print("at neuron "+b+":    ");
////				for (int j = 0; j < Constants.attributesIndexes.get(b).length; j++) {
////					System.out.print(Constants.attributesIndexes.get(b)[j]+"  ");
////				}
////				System.out.println();
//
//				Instances  incs = createProperDataset(z, true);
//				J48 btree = new J48();
//				btree.buildClassifier(incs);
//				
//
//				Iterator<Instance> it = incs.iterator();
//				int correct = 0;
//				while (it.hasNext()) {
//
//					double[] prediciton;
//					try {
//						Instance next = it.next();
//						prediciton = btree.predicate(next, true);
//							
//						INDArray nd = Nd4j.create(prediciton);
//						System.out.println(nd.maxNumber());
//						int idx = Nd4j.getExecutioner().execAndReturn(new IAMax(nd)).getFinalResult();	
//						if ( idx == next.classValue())
//							correct ++;
////						else {
//////							System.out.println(idx);
////						}
//
//					} catch (Exception e) {
//						e.printStackTrace();
//					}
//				}
//
//				System.out.println(((double)correct) / test.size());
////				
				
				
//				model.fit(set);
				

//			}
		model.fit(trainingData);

			
			
			if ( i % 2 == 0) {
				Constants.isEvaluating = true;
				Evaluation eval = new Evaluation(outputNum);
				System.out.println(testData.numExamples());
				INDArray output = model.output(testData.getFeatures());
				System.out.println("sdaf");
				eval.eval(testData.getLabels(), output);
				
				 String path = "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/Not Incremental/Epoch Iteration/resultIteration_"+ i;
				  File file = new File (path);
				  BufferedWriter out = new BufferedWriter(new FileWriter(file)); 
				  out.write(eval.stats()+"\n" + model.score());
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