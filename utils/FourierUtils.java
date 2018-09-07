package utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.complex.Complex;

import neuralnetwork.ActivationFunction;
import weka.classifiers.trees.ht.HNode;
import weka.classifiers.trees.ht.SplitNode;

public class FourierUtils {

	public static HashMap<ArrayList<String>, Complex> getDerivativeWeights(ActivationFunction activation,
			ArrayList<String> x) throws Exception {

		HashMap<ArrayList<String>, Complex> derivativeWeight = new HashMap<>();
		for (ArrayList<String> j : activation.weights.keySet()) {

			Complex bias = getFourierBasis(x, j, activation);

			double temp = 0;

			for (int i = 0; i < j.size(); i++)
				temp += (1 / activation.attCardinality.get(i)) * (Double.parseDouble(j.get(i)));

			derivativeWeight.put(j, bias.multiply(2 * Math.PI * temp));

		}

		return derivativeWeight;
	}

	private static void ExtractFS(Set<ArrayList<String>> S, HNode node, ArrayList<String> h,
			ActivationFunction activation) throws Exception {

		int xk = _utils.getSplittingFeature(node);
		Object[] children = ((SplitNode) node).m_children.values().toArray();
		Set<ArrayList<String>> N = _utils.XOR(S, _utils.delta(xk, activation));
		int k = xk;
		double size = ((double) _utils.getSpace(h, activation))
				/ (activation.attCardinality.get(xk) * _utils.getCompleteInstanceSpace(activation));

		for (ArrayList<String> j : N) {

			if (!activation.weights.containsKey(j))
				activation.weights.put(j, new Complex(0, 0));

			Complex weight_j = activation.weights.get(j);
			// for (int i = 0; i < Constants.attCardinality.get(k); i++) {
			for (int i = 0; i < ((SplitNode) node).numChildred(); i++) {

				ArrayList<String> hi = _utils.crossUnion(h, k, i);
				Complex fb = getFourierBasis(hi, j, activation);
				Double f = activation.Fxk.get(children[i]);
				Complex temp = fb.multiply(f);

				weight_j = weight_j.add(temp.multiply(size));
			}
			activation.weights.put(j, weight_j);

		}

		S.addAll(N);
		int i = 0;

		for (String key : ((SplitNode) node).m_children.keySet()) {
			ArrayList<String> hi = _utils.crossUnion(h, k, i);
			HNode child = ((SplitNode) node).m_children.get(key);
			if (child.isLeaf() == false)
				ExtractFS(S, ((SplitNode) node).m_children.get(key), hi, activation);
			i++;
		}

	}

	public static void setFourierSeriesWeights(HNode root, ActivationFunction activation) throws Exception {
		Set<ArrayList<String>> S = new HashSet<>();
		ArrayList<String> root_s = new ArrayList<>();
		ArrayList<String> j = new ArrayList<>();
		ArrayList<String> h = new ArrayList<>();

		for (int i = 0; i < activation.attCardinality.size(); i++) {
			root_s.add("0");
			j.add("0");
			h.add("*");
		}
		S.add(root_s);
		initF_xk(root, activation);
		activation.weights = new HashMap<>();
		activation.weights.put(j, new Complex(activation.Fxk.get(root), 0));
		if (root.isLeaf() == false)
			ExtractFS(S, root, h, activation);

	}

	private static Complex getFourierBasis(ArrayList<String> x, ArrayList<String> j, ActivationFunction activation)
			throws Exception {
		Complex i = new Complex(0, 1);
		Complex result = new Complex(0, 0);

		List<ArrayList<String>> X = convertX(x, activation);

		for (ArrayList<String> x_revised : X) {
			Complex temp1 = new Complex(1, 0);

			if (x.size() != j.size())
				throw new Exception("wrong inputs");
			int l = x_revised.size();
			for (int m = 1; m < l; m++) {
				double lambdam = activation.attCardinality.get(m);
				Complex imaginaryPart = (i.multiply(2 * Math.PI)).divide(new Complex(lambdam, 0));
				Complex power = imaginaryPart.multiply(new Complex(Double.parseDouble(x_revised.get(m)), 0));
				power = power.multiply(new Complex(Double.parseDouble(j.get(m)), 0));
				Complex t = power.exp();
				temp1 = t.multiply(temp1);

			}

			result = result.add(temp1);
		}

		return result.divide(new Complex(X.size(), 0));
	}

	private static void initF_xk(HNode node, ActivationFunction activation) {

		double factor = 0d;
		factor = getAvgOutput(node);

		if (node.isLeaf() == true) {
			activation.Fxk.put(node, factor);
			return;
		}

		Object[] children = ((SplitNode) node).m_children.values().toArray();
		double avg = 0d;

		for (int i = 0; i < children.length; i++)
			initF_xk((HNode) children[i], activation);

		for (int i = 0; i < children.length; i++)
			avg += activation.Fxk.get(children[i]);

		avg = avg * factor;
		activation.Fxk.put(node, avg);

	}

	private static double getAvgOutput(HNode node) {
		double avg = 0d;
		int counter = 0;
		for (String c : node.m_classDistribution.keySet()) {
			if (node.m_classDistribution.get(c).m_weight > 1) {
				avg += Integer.parseInt(c);
				counter++;
			} else {
				System.out.println("shitshitshitshitshit");
				System.out.println(node.m_classDistribution.get(c).m_weight);
			}
		}
		return avg / counter;

	}

	@SuppressWarnings("unchecked")
	private static List<ArrayList<String>> convertX(ArrayList<String> x, ActivationFunction activation) {

		ArrayList<ArrayList<String>> result = new ArrayList<>();
		if (x.contains("*") == false) {
			result.add(x);
			return result;
		}

		for (int i = 0; i < x.size(); i++) {
			if (x.get(i).compareTo("*") == 0) {
				for (int j = 0; j < activation.attCardinality.get(i); j++) {
					ArrayList<String> temp = (ArrayList<String>) x.clone();
					temp.set(i, j + "");
					result.addAll(convertX(temp, activation));
				}
			}

		}

		return result;
	}
}
