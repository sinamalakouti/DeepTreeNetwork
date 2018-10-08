package utils;

import java.util.ArrayList;
import java.util.HashMap;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import weka.core.Instances;

public class Constants {

//	TODO  : these are unique for each  neuron
//	public static ArrayList<Integer> attCardinality = new ArrayList<>();
//	public static int completeInstanceSpace = 0;
//	public static HashMap<HNode, Double> Fxk = new HashMap<>();
//	public static HashMap<ArrayList<String> , Complex > weights ;
//	public static List<ArrayList<Double>> cutpoints = new ArrayList<>();
	
	public static Instances train;
	public static Instances test;
	public static INDArray trainInstancesLabel;
	public static INDArray testInstancesLabel;
	public static boolean isEvaluating = false;
	public static int maximumDepth;
	public static HashMap<Integer,ArrayList<Integer>> classChosedArray = new HashMap<>();
	public static HashMap<Integer,int[]> attributesIndexes = new HashMap<>();
	public static	MultiLayerNetwork model;
	public static double [] weightLayerMin;
	public static double [] weightLayerMax;
	public static int numClasses = 10;
	
	
	
}
