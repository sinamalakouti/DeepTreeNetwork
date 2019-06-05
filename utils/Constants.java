package utils;

import java.util.ArrayList;
import java.util.HashMap;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import neuralnetwork.HoeffdingTree.HoeffdingTreeActivationFunction;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instances;

public class Constants {

//	TODO  : these are unique for each  neuron
//	public static ArrayList<Integer> attCardinality = new ArrayList<>();
//	public static int completeInstanceSpace = 0;
//	public static HashMap<HNode, Double> Fxk = new HashMap<>();
//	public static HashMap<ArrayList<String> , Complex > weights ;
//	public static List<ArrayList<Double>> cutpoints = new ArrayList<>();
	
//	public static Instances train;
//	public static Instances test;
	public static INDArray trainInstancesLabel;
	public static INDArray testInstancesLabel;
	public static boolean isEvaluating = false;
	public static int maximumDepth ;
	public static int numberOfLayers;
	public static int numberOfNeurons;
	public static int batchSize;
	public static int numBatches; 
	public static HashMap<Integer,ArrayList<Integer>> classChosedArray = new HashMap<>();
	public static HashMap<Integer,int[]> attributesIndexes = new HashMap<>();
	public static ArrayList<HashMap<Integer,int[]>> attributesIndexes2 = new ArrayList<>();
	public static MultiLayerNetwork model;
	public static boolean isSerialzing = false;
	public static boolean isDeSerializing = false;
	
	public static double [] weightLayerMin;
	public static double [] weightLayerMax;
	public static int numClasses ;
	public static double [] avgHFDepth;
	
	public static boolean  isCompare =false;
	public static ArrayList<HashMap<Integer, HoeffdingTree>>  trees2;
	public static ArrayList<HashMap<Integer, HoeffdingTreeActivationFunction>>  trees1;
	public static HashMap<Integer, ArrayList<Integer>> dropout = new HashMap<>();
	public static double dropoutRate;
	public static boolean isDropoutEnable  = false;
	public static String output_file_prefix = "";
	
	
}
