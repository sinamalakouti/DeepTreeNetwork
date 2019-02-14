package controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import neuralnetwork.CustomLayer;
import utils.Constants;
import utils._utils;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * Handwritten digits image classification on MNIST dataset (99% accuracy).
 * This example will download 15 Mb of data on the first run.
 * Supervised learning best modeled by CNN.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 */
public class Conv_Test {

  private static final Logger log = LoggerFactory.getLogger(Conv_Test.class);
  private static final String basePath = System.getProperty("java.io.tmpdir") + "/mnist";
  private static final String dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  public static void main(String[] args) throws Exception {
    int height = 28;
    int width = 28;
    int channels = 1; // single channel for grayscale images
    int outputNum = 10; // 10 digits classification
    int batchSize = 54;
    int nEpochs = 1;
    int iterations = 1;

    int seed = 1234;
    Random randNumGen = new Random(seed);

    log.info("Data load and vectorization...");
//    String localFilePath = basePath + "/mnist_png.tar.gz";
//    if (DataUtilities.downloadFile(dataUrl, localFilePath))
//      log.debug("Data downloaded from {}", dataUrl);
//    if (!new File(basePath + "/mnist_png").exists())
//      DataUtilities.extractTarGz(localFilePath, basePath);

    // vectorization of train data
//    File trainData = new File(basePath + "/mnist_png/training");
//    FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
//    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
//    trainRR.initialize(trainSplit);
//    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

    // pixel values from 0-255 to 0-1 (min-max scaling)
//    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//    scaler.fit(trainIter);
//    trainIter.setPreProcessor(scaler);

//    // vectorization of test data
//    File testData = new File(basePath + "/mnist_png/testing");
//    FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//    ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
//    testRR.initialize(testSplit);
//    DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
//    testIter.setPreProcessor(scaler); // same normalization for better results

    
    Constants.numberOfLayers = 2;
	Constants.numberOfNeurons = 40;
	Constants.batchSize = 100;
	Constants.avgHFDepth = new double[Constants.numberOfLayers];
	double numberTrainExamples = 60000d;
	Constants.numBatches = (int) ((numberTrainExamples) / Constants.batchSize);
	Constants.numClasses = 10;
	Constants.maximumDepth = 20;
	
	ArrayList<Integer> tmp1 = new ArrayList<Integer>();

	
	while (tmp1.size() < Constants.numberOfNeurons)
		tmp1.add(Constants.numClasses - 1);

	for (int l = 0; l < Constants.numberOfLayers; l++) {

		@SuppressWarnings("unchecked")
		ArrayList<Integer> tmp2 = (ArrayList<Integer>) tmp1.clone();
		Collections.shuffle(tmp2);
		Constants.classChosedArray.put(l, tmp2);
	}

	

    DataSetIterator mnistTrain = new MnistDataSetIterator(100, true, 6);
	DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 6);

    log.info("Network configuration and training...");
//    Map<Integer, Double> lrSchedule = new HashMap<>();
//    lrSchedule.put(0, 0.06); // iteration #, learning rate
//    lrSchedule.put(200, 0.05);
//    lrSchedule.put(600, 0.028);
//    lrSchedule.put(800, 0.0060);
//    lrSchedule.put(1000, 0.001);

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
		.trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
        .l2(0.0005)
        .updater(new Sgd(0.01))
        .weightInit(WeightInit.XAVIER)
        .list()
        .layer(0, new ConvolutionLayer.Builder(6, 6)
            .nIn(channels)
            .stride(3, 3)
            .nOut(1)
            .activation(Activation.IDENTITY)
            .build())

		.layer(1,
				new CustomLayer.Builder().nOut(Constants.numberOfNeurons)
						.activation(Activation.SIGMOID).build())
		.layer(2,
				new CustomLayer.Builder().nIn(Constants.numberOfNeurons).nOut(Constants.numberOfNeurons)
						.activation(Activation.SIGMOID).build())
        
        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(outputNum)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutionalFlat(28, 28, 1)) // InputType.convolutional for normal image
        .backprop(true).pretrain(false).build();

     Constants.model = new MultiLayerNetwork(conf);
     Constants.model.init();
     Constants.model.setListeners(new ScoreIterationListener(10));
    log.debug("Total num of params: {}", Constants.model.numParams());

    // evaluation while training (the score should go down)
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
	NumericToNominal convert = new NumericToNominal();
	String[] options = new String[2];
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
	HoeffdingTree batchTree = null;
	HoeffdingTree[] baggingTrees = new HoeffdingTree[Constants.numberOfNeurons];
	for (int i = 0; i < 1000; i++) {
		// in the first iteration do the bagging test and the each batch
		// test :D
		for (int b = 0; b < Constants.numBatches; b++) {
//			System.out.println(Constants.numBatches);

			DataSet set = getBatchTrainSet(b, Constants.batchSize, tempTrainSet, trainSet2);
			// if (i == 0) {
			//
			// Instances batchTrainInstance = _utils.dataset2Instances(set);
			//
			// // batch test :
			//
			// if (batchTree == null) {
			// batchTree = new HoeffdingTree();
			// batchTree.buildClassifier(batchTrainInstance);
			// } else {
			// Iterator<Instance> it = batchTrainInstance.iterator();
			// while (it.hasNext())
			// batchTree.updateClassifier(it.next());
			// }
			//
			// // bagging test:
			// for (int t = 0; t < Constants.numberOfNeurons; t++) {
			// INDArray bag =
			// _utils.getSubDataset(Constants.attributesIndexes.get(t),
			// set);
			// Instances bagInstances = _utils.ndArrayToInstances(bag);
			// if (baggingTrees[t] == null) {
			// baggingTrees[t] = new HoeffdingTree();
			// baggingTrees[t].buildClassifier(bagInstances);
			// } else {
			// Iterator<Instance> it = bagInstances.iterator();
			// while (it.hasNext()) {
			// baggingTrees[t].updateClassifier(it.next());
			// }
			// }
			//
			// }
			//
			// }

			Constants.model.fit(set);
		}
			if (i % 2 == 0) {
				
				
					 Constants.isEvaluating = true;
					 log.info("Evaluate model....");
					
					 Evaluation eval = new Evaluation(outputNum); // create an
					
					 while (mnistTest.hasNext()) {
					
					 DataSet next = mnistTest.next();
					 System.out.println(Constants.isEvaluating);
					 _utils.setLabels(next.getLabels(), Constants.isEvaluating,
					 false);
					 INDArray output = Constants.model.output(next.getFeatures());
					
					 eval.eval(next.getLabels(), output);
					 }
					 mnistTest.reset();
					
					 String path =
					 "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/phase4/CNN/3/resultIteration_"+ i;
					// String path =
					// "resultIteration_"+ i;
					 File file = new File(path);
					 BufferedWriter out = new BufferedWriter(new FileWriter(file));
					 String avglayersTreesDepth = "";
					 for ( int l = 0 ; l<Constants.numberOfLayers; l++)
					 avglayersTreesDepth = avglayersTreesDepth + " " +
					 Constants.avgHFDepth[l];
					 out.write(eval.stats() + "\nerrors\t" + Constants.model.score() + "\n" + avglayersTreesDepth);

					 System.out.println(eval.stats() + "\n" + "errors:  "+
					 Constants.model.score() + "\n" + avglayersTreesDepth);
					
					
					 out.close();
					 Constants.isEvaluating = false;
					
					 }
	}
  
   

//    ModelSerializer.writeModel(net, new File(basePath + "/minist-model.zip"), true);
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
		features = null;
		labels = null;
		batchTrain_features = null;
		batchTrain_labels = null;
		return set;

	}

}