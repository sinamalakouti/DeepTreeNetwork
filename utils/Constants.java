package utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.complex.Complex;

import weka.classifiers.trees.ht.HNode;

public class Constants {

	
	public static ArrayList<Integer> attCardinality = new ArrayList<>(3);
	public static int completeInstanceSpace = 0;
	public static HashMap<HNode, Double> Fxk = new HashMap<>();
	public static HashMap<ArrayList<String> , Complex > weights ;
	public static List<ArrayList<Double>> cutpoints = new ArrayList<>();
	
}
