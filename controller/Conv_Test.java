package controller;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
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
import neuralnetwork.HoeffdingTree.HoeffdingTreeActivationFunction;
import utils.Constants;
import utils._utils;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.pmml.Constant;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * Handwritten digits image classification on MNIST dataset (99% accuracy).
 * This example will download 15 Mb of data on the first run.
 * Supervised learning best modeled by CNN.
 *
 * @author 	
 * @author agibsonccc
 * @author fvaleri
 */
public class Conv_Test {

  private static final Logger log = LoggerFactory.getLogger(Conv_Test.class);
  
  

  public static void main(String[] args) throws Exception {
    
    
    int channels = 1; // single channel for grayscale images
    

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

    
    Constants.numberOfLayers = 1 ;
	Constants.numberOfNeurons = 40;
	Constants.batchSize = 100;
	Constants.avgHFDepth = new double[Constants.numberOfLayers];
	double numberTrainExamples = 60000d;
	Constants.numBatches = (int) ((numberTrainExamples) / Constants.batchSize);
	Constants.setNumClasses(10);
	Constants.maximumDepth = 20;
	  Constants.base_hf_layerNumber =2;
	int outputnum =10;
	
	int numInputs =2880;
	ArrayList<Integer> featuresVector = new ArrayList<>();
	for (int i = 0; i < numInputs; i++)
		featuresVector.add(i);

	int max = numInputs / 30;
	HashMap<Integer, Boolean> attInexes = new HashMap<>();
	for (int j = 0; j < Constants.numberOfNeurons; j++) {
		Collections.shuffle(featuresVector);
		int[] temp = new int[max];
		for (int i = 0; i < max; i++) {
			temp[i] = featuresVector.get(i);
			attInexes.put(featuresVector.get(i), true);
		}

		Constants.attributesIndexes.put(j, temp);

	}
	
//
	ArrayList<Integer> tmp1 = new ArrayList<Integer>();


	for (int c = 0; c < Constants.getNumClasses() - 1; c++) {
		// for 4 classes -> it is set only for mnist dataset ( to be changed
		// )
		for (int i = 0; i < (int) (Constants.numberOfNeurons / Constants.getNumClasses()); i++) {
			tmp1.add(c);
		}
	}

	while (tmp1.size() < Constants.numberOfNeurons)
		tmp1.add(Constants.getNumClasses() - 1);

		for (int l = 0; l < Constants.numberOfLayers; l++) {
	
			@SuppressWarnings("unchecked")
			ArrayList<Integer> tmp2 = (ArrayList<Integer>) tmp1.clone();
			Collections.shuffle(tmp2);
			Constants.getClassChosedArray().put(l + Constants.base_hf_layerNumber, tmp2);
		}

	

    DataSetIterator mnistTrain = new MnistDataSetIterator(Constants.batchSize, true, 6);
	DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 6);
	



	    // pixel values from 0-255 to 0-1 (min-max scaling)
	    DataNormalization scaler = new NormalizerStandardize();
	    scaler.fit(mnistTrain);
	    mnistTrain.setPreProcessor(scaler);	 
	    mnistTest.setPreProcessor(scaler); // same normalization for better results

    log.info("Network configuration and training...");

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
		.trainingWorkspaceMode(WorkspaceMode.NONE).inferenceWorkspaceMode(WorkspaceMode.NONE)
        .l2(0.0001)
        .updater(new Sgd(0.1))
        .weightInit(WeightInit.XAVIER)
        .list()
        .layer(0, new ConvolutionLayer.Builder(5, 5)
                .nIn(channels)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.IDENTITY)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new CustomLayer.Builder().activation(new HoeffdingTreeActivationFunction(-1, false, -1))
					.nIn(numInputs)
                    .nOut(Constants.numberOfNeurons).build())


//            .layer(3, new CustomLayer.Builder().activation(new HoeffdingTreeActivationFunction(-1, false, -1))
//                    .nOut(Constants.numberOfNeurons).build())
//            .layer(3, new CustomLayer.Builder().activation(new HoeffdingTreeActivationFunction(-1, false, -1))
//                    .nOut(Constants.numberOfNeurons).build())
            
                .layer(3	, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                		.nOut(outputnum)
                        .activation(Activation.SOFTMAX)
                        .build())
        .setInputType(InputType.convolutionalFlat(28, 28, 1)) // InputType.convolutional for normal image
        .build();

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
					
					 Evaluation eval = new Evaluation(outputnum); // create an
					
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
					 "/home/sina/eclipse-workspace/ComplexNeuronsProject/result/phase4/CNN/6/resultIteration_"+ i;
//					 String path =
//					 "resultIteration_"+ i;
					 File file = new File(path);
					 BufferedWriter out = new BufferedWriter(new FileWriter(file));
					 String avglayersTreesDepth = "";
					 for ( int l = 0 ; l<Constants.numberOfLayers; l++)
					 avglayersTreesDepth = avglayersTreesDepth + " " +
					 Constants.avgHFDepth[l];
					 out.write(eval.stats() + "\nerrors\t" + Constants.model.score() + "\n" + avglayersTreesDepth);
//
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