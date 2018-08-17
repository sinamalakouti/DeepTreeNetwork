package utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.complex.Complex;

import weka.classifiers.trees.ht.HNode;
import weka.classifiers.trees.ht.SplitNode;

public class FourierUtils {


	private static void  ExtractFS( Set<ArrayList<String>> S, HNode node, ArrayList<String> h) throws Exception {


 		int xk = _utils.getSplittingFeature(node);
		Object[] children =   ((SplitNode)node).m_children.values().toArray();
		Set<ArrayList<String>> N = _utils.XOR(S, _utils.delta(xk));
		int k = xk;
		double size = ((double)_utils.getSpace(h))/  (Constants.attCardinality.get(xk) * _utils.getCompleteInstanceSpace());

		for (ArrayList<String> j : N) {
			
			if ( ! Constants.weights.containsKey(j))
				Constants.weights.put(j, new Complex(0, 0));

			Complex weight_j = Constants.weights.get(j);
//			for (int i = 0; i < Constants.attCardinality.get(k); i++) {
			for ( int i = 0 ; i < (  (SplitNode) node  ).numChildred() ; i ++ ) {

				ArrayList<String> hi =  _utils.crossUnion(h, k, i);
				Complex fb = getFourierBasis(hi, j);
				Double f =  Constants.Fxk.get(children[i]);
				Complex temp = fb.multiply(f);
				
				weight_j = weight_j.add(temp.multiply(size));
			}
			Constants.weights.put(j, weight_j);

		}



		S.addAll(N);
		int i = 0; 

		for (String key : ((SplitNode)node).m_children.keySet() ) {
						ArrayList<String> hi  = _utils.crossUnion(h, k, i);
			HNode child = ((SplitNode)node).m_children.get(key);
			if(child.isLeaf() == false)
				ExtractFS(S, ((SplitNode)node).m_children.get(key), hi);
			i++;
		}

	}

	public static void setFourierSeriesWeights(HNode root) throws Exception{
		Set<ArrayList<String>> S = new HashSet<>();
		ArrayList<String> root_s = new ArrayList<>();
		ArrayList<String> j = new ArrayList<>();
		ArrayList<String> h = new ArrayList<>();

		for ( int i =0 ; i< Constants.attCardinality.size(); i ++) {
			root_s.add("0");
			j.add("0");
			h.add("*");
		}
		S.add(root_s);
		initF_xk(root);
		Constants.weights = new HashMap<>();
		Constants.weights.put(j, new Complex(Constants.Fxk.get(root), 0));
		if (root.isLeaf() == false)
			ExtractFS(S, root, h);


	}

	private static Complex getFourierBasis( ArrayList<String> x , ArrayList<String> j) throws Exception {
		Complex i = new Complex(0, 1);
		Complex result = new Complex(0, 0);

		List<ArrayList<String>> X = convertX(x);
		
		
		for (ArrayList<String> x_revised : X) {
			Complex temp1 = new Complex(1, 0);

			if (x.size() != j.size()) 
				throw new Exception("wrong inputs");
			int l = x_revised.size();
			for (int m = 1; m < l; m ++) {
				double lambdam = Constants.attCardinality.get(m) ;
				Complex imaginaryPart = (i.multiply(2 * Math.PI)).divide(new Complex(lambdam, 0));
				Complex power = imaginaryPart.multiply( new Complex(Double.parseDouble(x_revised.get(m)) , 0));
				power = power.multiply(new Complex(Double.parseDouble(j.get(m)), 0));
				Complex t =  power.exp();
				temp1 = t.multiply(temp1);
				
			}
			
			result  = result.add(temp1);
		}
		

		
		return result.divide(new Complex(X.size(), 0));
	}

	private static void initF_xk( HNode node) {

		double factor = 0d;
		factor = getAvgOutput(node);


		if (node.isLeaf() == true) {
			Constants.Fxk.put(node, factor);
			return;
		}

		Object[] children =  ((SplitNode)node).m_children.values().toArray();
		double avg = 0d;

		for ( int i =0 ; i < children.length ; i++) 
			initF_xk( (HNode)children[i]);



		for ( int i = 0 ; i < children.length ; i++)
			avg += Constants.Fxk.get(children[i]);

		avg = avg * factor;
		Constants.Fxk.put(node, avg);

	}

	private static double getAvgOutput(HNode node) {
		double avg = 0d;
		int counter = 0;
		for (String c : node.m_classDistribution.keySet()) {
			if ( node.m_classDistribution.get(c).m_weight > 1) {
				avg += Integer.parseInt(c);
				counter ++;
			}else
			{
				System.out.println("shitshitshitshitshit");
				System.out.println( node.m_classDistribution.get(c).m_weight);
			}
		}
		return avg / counter;

	}


	@SuppressWarnings("unchecked")
	private static List<ArrayList<String>> convertX( ArrayList<String> x){
		
       		ArrayList<ArrayList<String>> result = new ArrayList<>();
		if(x.contains("*") == false) {
			 result.add(x);
			 return result;
		}
		
		
		for( int i =0 ; i< x.size() ; i++) {
			if ( x.get(i).compareTo("*") == 0) {
				for( int j = 0 ; j< Constants.attCardinality.get(i); j++) {
					ArrayList<String> temp = (ArrayList<String>) x.clone();
					temp.set(i, j +"" );
					result.addAll(  convertX(temp) );
				}
			}

		}
		
 		return result;
	}
}
