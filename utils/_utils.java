package utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import neuralnetwork.ActivationFunction;
import weka.classifiers.trees.ht.HNode;
import weka.classifiers.trees.ht.SplitNode;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class _utils {

	public static int[] getLabels(Instances data) {
		int[] list = new int[data.size()];
		data.setClassIndex(data.numAttributes() - 1);
		for (int i = 0; i < data.size(); i++) {
			list[i] = (int) data.get(i).classValue();
		}
		return list;
	}

	public static void setLabels(INDArray labels, boolean evaluating, boolean training) {

		INDArray idxOfMaxInEachColumn = Nd4j.getExecutioner().exec(new IAMax(labels), 1);

		if (training)
			Constants.trainInstancesLabel = idxOfMaxInEachColumn;
		else if (evaluating)
			Constants.testInstancesLabel = idxOfMaxInEachColumn;

	}

	public static Instances dataset2Instances(DataSet ds) throws Exception {

		INDArray features = ds.getFeatures();
		INDArray labels = ds.getLabels();
		INDArray idx_labels = Nd4j.getExecutioner().exec(new IAMax(labels), 1);
		INDArray features_labels = Nd4j.concat(1, features, idx_labels).dup();
		Instances instances = _utils.ndArrayToInstances(features_labels);
		NumericToNominal convert = new NumericToNominal();
		convert = new NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "" + (instances.numAttributes()); // range of variables to
		convert.setOptions(options);
		convert.setInputFormat(instances);
		instances = weka.filters.Filter.useFilter(instances, convert);
		instances.setClassIndex(instances.numAttributes()-1);
	

		return instances;
	}
	
	public static INDArray getSubDataset(int[] featursIndx , DataSet ds){
		INDArray features = ds.getFeatures().getColumns(featursIndx);
		INDArray labels = ds.getLabels();
		INDArray idx_labels = Nd4j.getExecutioner().exec(new IAMax(labels), 1);
		INDArray features_labels = Nd4j.concat(1, features, idx_labels).dup();
		
		return features_labels;
		
	}

	public static INDArray convertActivtionOutput(INDArray arg0, double[][] prediction) throws Exception {
		INDArray arr = Nd4j.zeros(prediction.length, prediction[0].length);
		arr = Nd4j.create(prediction);
		INDArray dataset = Nd4j.concat(1, arg0, arr);
		Instances instances = weka.classifiers.functions.dl4j.Utils.ndArrayToInstances(dataset);
		instances.setClassIndex(instances.numAttributes() - 1);

		NumericToNominal convert = new NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "" + (instances.classIndex() + 1); // range of variables to
		// make numeric
		convert.setOptions(options);
		convert.setInputFormat(instances);
		instances = weka.filters.Filter.useFilter(instances, convert);

		DataSet lables = weka.classifiers.functions.dl4j.Utils.instancesToDataSet(instances);
		return lables.getLabels();
	}

	public static int getSplittingFeature(HNode node) throws Exception {

		int attribute = -1;
		String att = ((SplitNode) node).getSplitAtt();
		int attNumeric = Integer.parseInt("" + att.charAt(att.length() - 1));
		attribute = attNumeric;

		if (attribute == -1) {
			throw new Exception(" cannot get split attribute of leaf node");
		}

		return attribute;

	}

	public static Set<ArrayList<String>> XOR(Set<ArrayList<String>> S, HashMap<Integer, ArrayList<Integer>> sigma_xk) {
		HashSet<ArrayList<String>> result = new HashSet<>();
		for (ArrayList<String> partition : S) {
			for (Integer key : sigma_xk.keySet()) {

				for (int i = 0; i < sigma_xk.get(key).size(); i++) {

					@SuppressWarnings("unchecked")
					ArrayList<String> temp = crossUnion((ArrayList<String>) partition.clone(), key,
							sigma_xk.get(key).get(i));
					result.add(temp);
				}

			}
		}
		return result;
	}

	public static HashMap<Integer, ArrayList<Integer>> delta(int xk, ActivationFunction activation) {

		HashMap<Integer, ArrayList<Integer>> result = new HashMap<>();

		for (int i = 1; i < activation.attCardinality.get(xk); i++) {
			if (!result.containsKey(xk)) {
				ArrayList<Integer> t = new ArrayList<>();
				t.add(i);
				result.put(xk, t);
			} else {
				ArrayList<Integer> t = result.get(xk);
				t.add(i);
				result.put(xk, t);
			}

		}

		return result;

	}

	public static ArrayList<String> crossUnion(ArrayList<String> h, int attribute, int value) {
		h.set(attribute, value + "");

		return h;

	}

	public static int getNodeSize(HNode node) {
		return -1;
	}

	public static int getSpace(ArrayList<String> nodeSchema, ActivationFunction activation) {
		int size = 1;
		for (int i = 0; i < nodeSchema.size(); i++) {
			if (nodeSchema.get(i).compareTo("*") == 0) {
				size *= activation.attCardinality.get(i);
			}
		}
		return size;
	}

	public static double getCompleteInstanceSpace(ActivationFunction activation) {
		if (activation.completeInstanceSpace == 0) {
			activation.completeInstanceSpace = 1;
			for (int i = 0; i < activation.attCardinality.size(); i++) {
				if (activation.attCardinality.get(i) != 0)
					activation.completeInstanceSpace *= activation.attCardinality.get(i);
			}
		}

		return activation.completeInstanceSpace;

	}

	public static void setAttCardinality(ArrayList<Double>[] cutpoints, ActivationFunction activation) {
		for (int i = 0; i < cutpoints.length; i++) {
			if (cutpoints[i] != null) {
				if (i >= activation.attCardinality.size())
					activation.attCardinality.add(0);
				activation.attCardinality.set(i, cutpoints[i].size() + 1);
			} else if (i >= activation.attCardinality.size())
				activation.attCardinality.add(0);
			// else
			// Constants.attCardinality.add(0);
			//

		}

	}

	public static DataSet instancesToDataSet(Instances insts) {
		INDArray data = Nd4j.zeros(insts.numInstances(), insts.numAttributes() - 1);
		INDArray outcomes = Nd4j.zeros(insts.numInstances(), insts.numClasses());

		for (int i = 0; i < insts.numInstances(); i++) {
			double[] independent = new double[insts.numAttributes() - 1];
			double[] dependent = new double[insts.numClasses()];
			Instance current = insts.instance(i);
			for (int j = 0; j < current.numValues(); j++) {
				int index = current.index(j);
				double value = current.valueSparse(j);

				if (index < insts.classIndex()) {
					independent[index] = value;
				} else if (index > insts.classIndex()) {
					// Shift by -1, since the class is left out from the feature
					// matrix and put into a separate
					// outcomes matrix
					independent[index - 1] = value;
				}
			}

			// Set class values
			if (insts.numClasses() > 1) { // Classification
				final int oneHotIdx = (int) current.classValue();
				dependent[oneHotIdx] = 1.0;
			} else { // Regression (currently only single class)
				dependent[0] = current.classValue();
			}

			INDArray row = Nd4j.create(independent);
			data.putRow(i, row);
			outcomes.putRow(i, Nd4j.create(dependent));
		}
		return new DataSet(data, outcomes);
	}

	public static Instances ndArrayToInstances(INDArray ndArray) throws Exception {

		// NDArray nd = new NDArray();
		// nd.std(1,2);
		long batchsize = (int) ndArray.size(0);
		long[] shape = ndArray.shape();
		int dims = shape.length;
		if (dims < 2) {
			throw new WekaException("Invalid input, NDArray shape needs to be at least two dimensional " + "but was "
					+ Arrays.toString(shape));
		}

		long prod = Arrays.stream(shape).reduce(1, (left, right) -> left * right);
		prod = prod / batchsize;

		ArrayList<Attribute> atts = new ArrayList<>();
		for (int i = 0; i < prod; i++) {
			atts.add(new Attribute("transformedAttribute" + i));
		}
		Instances instances = new Instances("Transformed", atts, (int) batchsize);
		for (int i = 0; i < batchsize; i++) {
			INDArray row = ndArray.getRow(i);
			INDArray flattenedRow = Nd4j.toFlattened(row);
			Instance inst = new DenseInstance(atts.size());
			for (int j = 0; j < flattenedRow.size(1); j++) {

				inst.setValue(j, flattenedRow.getDouble(j));
			}
			inst.setDataset(instances);
			instances.add(inst);
		}

		
		NumericToNominal convert = new NumericToNominal();
		convert = new NumericToNominal();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "" + (instances.numAttributes()); // range of variables to
		convert.setOptions(options);
		convert.setInputFormat(instances);
		instances = weka.filters.Filter.useFilter(instances, convert);
		instances.setClassIndex(instances.numAttributes()-1);
		
		return instances;
	}

	public static double calcPooledMean(double mu1, double n1, double mu2, double n2) {
		double temp = mu1 * n1 + mu2 * n2;
		return temp / (n1 + n2);
	}

	public static double calcPooledSTD(double std1, double n1, double std2, double n2) {

		double temp = (n1 - 1) * Math.pow(std1, 2) + (n2 - 1) * Math.pow(std2, 2);
		temp /= (n1 + n2 - 2);
		temp = Math.sqrt(temp);
		if (Double.isInfinite(Math.pow(std2, 2))) {
			System.out.println("in utils");
			System.out.println("ehe he");
			System.exit(0);
		}

		if (Double.isInfinite(temp)) {
			System.out.println("in utils");
			System.out.println("haaaa?");
			System.exit(0);
		}
		return Math.max(0.00001, temp);
	}
	
//	public static void draw_accuracy_fscore(String name, String path, int first, int last) throws FileNotFoundException{
//		
//		JGnuplot jg = new JGnuplot();
//		Plot plot = new Plot(name);
//		int n = ( last - first) / 2 + 1;
//		double[] x = new double [n];
//		double[][] fscore = new double[n][2];
//		double[][] accuracy = new double[n][2];
//		double [][] score = new double[n] [2];
//		int counter = 0;
//		for ( int i = 0 ; i < n ; i ++ ){
//			counter = (i * 2);
//			x[i] = counter + 1;
//			
//			File file = new File(path +"/resultIteration_" + i );
//			Scanner scan = new Scanner(file);
//			String line = null;
//			while (scan.hasNextLine()){
//			    line = scan.nextLine();
////		        String delim = "[^a-zA-Z1-9.]+";
//				String delim = "\\S+";
//				String [] tokens = line.split(delim);
//				if ( tokens[0].toLowerCase().compareTo("accuracy:") == 0){
//					accuracy[i][1] = counter+1;	
//				 accuracy[i][1] = Double.parseDouble(tokens[1]);
//				}else if ( tokens[0].toLowerCase().compareTo("f1") == 0){
//					fscore[i][0] = counter +1;
//					fscore[i][1] = Double.parseDouble(tokens[2]);
//				}
//				else 
//					continue;
//				
//						
//			}
//			score[i][0] = counter +1;
//			score[i][1]= Double.parseDouble(line);
//			
//		}
//		DataTableSet dts = plot.addNewDataTableSet("");
//		
//		JavaPlot jp = new JavaPlot();
//		jp.addPlot(accuracy);
//		jp.addPlot(fscore);
//		jp.addPlot(score);
//		jp.plot();
////		dts.addNewDataTable("sadfs",x);
////		dts.addNewDataTable("f1score", x, y2);
////		dts.addNewDataTable("f1score", x, y2);
////		jg.execute(plot, jg.plot2d);
//
//			
//	}

}
