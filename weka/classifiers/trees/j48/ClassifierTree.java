/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ClassifierTree.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.j48;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.Queue;

import utils.Constants;
import utils._utils;
import weka.classifiers.trees.ht.LeafNode;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * Class for handling a tree structure used for classification.
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 14534 $
 */
public class ClassifierTree implements Drawable, Serializable, RevisionHandler, CapabilitiesHandler {

	/** for serialization */
	static final long serialVersionUID = -8722249377542734193L;

	/** The model selection method. */
	protected ModelSelection m_toSelectModel;

	/** Local model at node. */
	protected ClassifierSplitModel m_localModel;

	/** References to sons. */
	protected ClassifierTree[] m_sons;

	/** True if node is leaf. */
	protected boolean m_isLeaf;

	/** True if node is empty. */
	protected boolean m_isEmpty;

	/** The training instances. */
	protected Instances m_train;

	/** The pruning instances. */
	protected Distribution m_test;

	/** The id for the node. */
	protected int m_id;

	/**
	 * For getting a unique ID when outputting the tree (hashcode isn't guaranteed
	 * unique)
	 */
	private static long PRINTED_NODES = 0;

//  Add by Sina

	protected double[][] mu;
	
	protected double[][] sd;
	protected double[] classProb;
	public int depth = 0;
	public int numInstances;

	public double[][] getMu() {
		return this.mu;
	}

	public double[][] getSd() {
		return this.sd;
	}

	public double[] getClassProb() {
		return this.classProb;
	}


	public double[] predicateDerviativ(Instance instance, boolean isOutputLayerActivation) throws Exception {

		return predicateDerivative_log(instance, isOutputLayerActivation);
	}

	public double[] predicateDerivative_log(Instance instance, boolean isOutputLayerActivation) throws Exception {

		
		ClassifierTree node = getBayesParameters(instance);

		double[] pdf_log = getGaussianPDF_log(instance, node.classProb, node.mu, node.sd, node.m_train);
		double[] gradient = new double[pdf_log.length];

// calculate ak_derivaitive		
		for (int c = 0; c < pdf_log.length; c++) {

			double temp = 0d;
			for (int a = 0; a < instance.numAttributes(); a++) {
				if (!instance.attribute(a).equals(instance.classAttribute()))
					temp += (node.mu[a][c] - instance.value(a)) / (Math.pow(node.sd[a][c], 2));
			}
			gradient[c] = temp;

		}

//		calculate the maximum pdf_log
		double maxPDF_log = Double.NEGATIVE_INFINITY;
		int maxPDF_LogIndex = -1;

		for (int i = 0; i < pdf_log.length; i++) {

			if (pdf_log[i] > maxPDF_log) {
				maxPDF_LogIndex = i;
				maxPDF_log = pdf_log[i];
			}
		}
//		calculate probability
		double sumPDF_log = 0;
		for (int i = 0; i < pdf_log.length; i++) {

			sumPDF_log += Math.exp(pdf_log[i] - maxPDF_log);

		}
		sumPDF_log = Math.log(sumPDF_log);
		sumPDF_log += maxPDF_log;

		double[] prob = new double[pdf_log.length];

		for (int i = 0; i < pdf_log.length; i++) {

			double temp = pdf_log[i] - sumPDF_log;
			prob[i] = Math.exp(temp);
			if (prob[i] == 0d)
				prob[i] = 0.00001d;
//			if ( prob[i] == 1)
//				prob[i] -= 0.00001d;

		}

		if (maxPDF_LogIndex == -1) {
			System.out.println("idex == -1 ajaaaab");
			System.exit(0);
		}

//		 now we need to calculate the derivative part		

		double denominator = 0d;
		double numerator = 0d;

		for (int c = 0; c < pdf_log.length; c++) {

			denominator += Math.exp(pdf_log[c] - maxPDF_log);
			numerator += (gradient[c] - gradient[maxPDF_LogIndex]) * Math.exp(pdf_log[c] - maxPDF_log);
		}

		double[] result = new double[pdf_log.length];

		for (int c = 0; c < pdf_log.length; c++) {
			result[c] = gradient[c] - (gradient[maxPDF_LogIndex] + (numerator / denominator));
			result[c] *= prob[c];

		}

		if (isOutputLayerActivation == true) {
			return result;
		}

		double[] output = new double[2];
		output[0] = result[maxPDF_LogIndex];
		output[1] = maxPDF_LogIndex;

		return output;
	}

	public double[] predicate(Instance instance, boolean isOutoutLayer) throws Exception {

		double[] currentPDFs_log;

		double[] maxProb_log = new double[2];
		double sumPDF_log = 0d;
		ClassifierTree node = getBayesParameters(instance);

		if (node.isLeaf() == false) {
			System.out.println("not a leaf");
			System.exit(0);
			
		}

		currentPDFs_log = getGaussianPDF_log(instance, node.classProb, node.mu, node.sd, node.m_train);

		double maxPDF_log = Double.NEGATIVE_INFINITY;

		int maxIndex = -1;

		for (int i = 0; i < currentPDFs_log.length; i++) {

			if (currentPDFs_log[i] > maxPDF_log) {
				maxIndex = i;
				maxPDF_log = currentPDFs_log[i];
			}
		}

		sumPDF_log = 0;
		for (int i = 0; i < currentPDFs_log.length; i++) {

			sumPDF_log += Math.exp(currentPDFs_log[i] - maxPDF_log);

		}
		if (sumPDF_log == 0d) {
			System.out.println("bakhtim akhavi");
			System.exit(0);
		}
		sumPDF_log = Math.log(sumPDF_log);
		sumPDF_log += maxPDF_log;

		double[] prob = new double[currentPDFs_log.length];

		double max = Double.NEGATIVE_INFINITY;
		maxIndex = -1;
		for (int i = 0; i < currentPDFs_log.length; i++) {

			double temp = currentPDFs_log[i] - sumPDF_log;
			prob[i] = Math.exp(temp);
			if (prob[i] == 0)
				prob[i] += 0.0000001d;
			if (temp > max) {
				max = temp;
				maxIndex = i;
			}
		}
		if (isOutoutLayer == true) {

			return prob;

		} else {
			
			if ( maxIndex == -1)
			{
				System.out.println("midooni chara");
				System.out.println("ya ali");
				
			}

			maxProb_log[0] = prob[maxIndex];
			maxProb_log[1] = maxIndex;
			if (Double.isNaN(maxProb_log[0]) || maxProb_log[0] > 1 || maxProb_log[0] < 0) {

				System.out.println("ya hossein");
				System.out.println(instance.classValue());
				for (int i = 0; i < instance.numAttributes(); i++)
					System.out.println(instance.value(i));
				currentPDFs_log = getGaussianPDF_log(instance, node.classProb, node.mu, node.sd, node.m_train);

				System.exit(0);
			}
			double sum = 0d;
			for (int i = 0; i < prob.length; i++) {
				sum += prob[i];
			}
			if (sum - 1 > 0.01 || ((1 - sum) > 0.01)) {
				System.out.println(sum);
				System.out.println("oza kharaaabbbeee");
				System.exit(0);
			}
			return maxProb_log;

		}

	}

	private ClassifierTree getBayesParameters(Instance instance) throws Exception {

		if (m_isLeaf) {
			// return weight * localModel().classProb(classIndex, instance, -1);
			if (mu == null) {
				System.out.println(m_train.size());
				System.out.println(m_train.numAttributes());
				System.out.println("khaaak");
				System.exit(0);
			}
			return this;
		} else {
			int treeIndex = localModel().whichSubset(instance);
			if (treeIndex == -1) {

				System.out.println("treeIndex problem ");
				System.exit(0);
				return this;

			} else {
				if (son(treeIndex).m_isEmpty) {
					if (this.isLeaf() == false) {
						System.out.println("leaf ghlaaaate");
						System.exit(0);
					}
					return this;
				} else {
					return son(treeIndex).getBayesParameters(instance);
				}
			}
		}
	}


	public double[] getGaussianPDF_log(Instance data, double[] classProb, double[][] sampleMeans,
			double[][] sampleStDevs, Instances dataset) {

		double[] pdf = new double[Constants.numClasses];
		if (sampleMeans.length != sampleStDevs.length || sampleMeans[0].length != sampleStDevs[0].length) {
			System.out.println("ERRRROOOOORRRR !!");
			System.exit(0);
		}				

		int n_attributes = sampleMeans.length;
		int n_classes = Constants.numClasses;

		Instance x = data;
		for (int c = 0; c < n_classes; c++) {
			double res = 0d;

			for (int i = 0; i < n_attributes; i++) {

				Attribute attribute = data.attribute(i);
				

				if ( sampleStDevs[i][c] == 0)
					sampleStDevs[i][c] = 0.1;
				
				double tempLog1 = Math.log(1 / (Math.sqrt(2 * Math.PI) * sampleStDevs[i][c]))
						- (Math.pow((x.value(attribute) - sampleMeans[i][c]), 2)
								/ (2 * Math.pow(sampleStDevs[i][c], 2)));

				if (Double.isNaN(tempLog1)) {
					System.out.println(Math.log(1 / (Math.sqrt(2 * Math.PI))));
					System.out.println(  (x.value(attribute) - sampleMeans[i][c])  );
					System.out.println(x.value(attribute));
					System.out.println( sampleMeans[i][c]);
					System.out.println("oh my god log log");
					System.exit(0);
				}
				res = res + tempLog1;
				if ( Double.isInfinite(tempLog1))
				{
					System.out.println("TEMPlOG1 IS INFINITE IN CLASSIFFIER");
				}
			}
			
			if (classProb[c] == 0)
				classProb[c] += 0.0001;
			res += Math.log(classProb[c]);
			if ( Double.isInfinite(res))
			{
				System.out.println("shitshit ya shitoone");
			}
			pdf[c] = res;
			if (Double.isNaN(pdf[c])) {
				System.out.println(classProb[c]);
				System.out.println("NAN again!!");
				System.exit(0);
			}

		}

		return pdf;

	}

	public double[] getPDFDerivative_log(Instance data, double[] classProb, double[][] sampleMeans,
			double[][] sampleSTDs) {

		int n_attributes = sampleMeans.length;
		int n_classes = sampleMeans[0].length;
		
		double[] grediant = new double[n_classes];
		for (int c = 0; c < n_classes; c++) {

			double derivativePart = 0d;
			
			for (int i = 0; i < n_attributes; i++) {
				Attribute attribute = data.attribute(i);
				if ( sampleSTDs[i][c] == 0)
					sampleSTDs[i][c] = 0.1;
				derivativePart += (data.value(attribute) - sampleMeans[i][c]) / (Math.pow(sampleSTDs[i][c], 2));

			}

			grediant[c] = -1 * derivativePart;
		}

		return grediant;
	}

	
//	
	
	  
	  public void update ( Instances data) throws Exception {
		
		  this.setParameters(data);
	  }
	protected void setParameters(Instances data) throws Exception {
		double[][] mu2 = new double[data.numAttributes() - 1][Constants.numClasses];
		double[][] sd2 = new double[data.numAttributes() - 1][Constants.numClasses];
		int numInstances2 = data.size();
		double[] classProb2 = new double[Constants.numClasses];

		Instances[] tempInstances = new Instances[Constants.numClasses];
		double s = 0d;
		for (int j = 0; j < data.classAttribute().numValues(); j++) {
			
			int c = Integer.parseInt(data.classAttribute().value(j));
			
			RemoveWithValues rwv = new RemoveWithValues();

			String[] options = new String[5];
			options[0] = "-C";
			options[1] = "" + (data.numAttributes());
			options[2] = "-L";
			options[3] = "" + (j + 1);
			options[4] = "-V";
			rwv.setOptions(options);
			rwv.setInputFormat(data);
			Instances xt = Filter.useFilter(data, rwv);
			tempInstances[c] = xt;

			if (data.size() == 0) {
				classProb2[c] = 0;
			} else
				classProb2[c] = ((double) xt.size()) / ((double) data.size());

			s += classProb2[c];
		}

		if (s - 1 > 0.001 || 1 - s > 0.001) {
			System.out.println("class problem!!");
		}

		for (int i = 0; i < data.numAttributes(); i++) {

			if (!data.attribute(i).equals(data.classAttribute())) {

				for (int j = 0; j < data.classAttribute().numValues(); j++) {

					int c = Integer.parseInt(data.classAttribute().value(j));

					if (tempInstances[c].size() < 2) {
						mu2[i][c] = 0;
						sd2[i][c] = 0.01;
					} else {
						try {
							mu2[i][c] = tempInstances[c].meanOrMode(i);
						} catch (Exception e) {
							System.out.println(data.classIndex());
							System.out.println("hereererererererer");
							System.out.println(tempInstances[c].attribute(i).isNominal());
							System.out.println(tempInstances[c].attributeStats(i).numericStats);
							System.exit(0);
						}
						sd2[i][c] = tempInstances[c].attributeStats(i).numericStats.stdDev + 0.1;
						if ( sd2[i][c] > 10) {
//							System.out.println("hallo");
						}
					}


				}

			}

		}

		if (mu == null && sd == null) {
			this.mu = mu2;
			this.numInstances = numInstances2;
			this.classProb = classProb2;
			this.sd = sd2;
		} else {

			for (int j = 0; j < data.classAttribute().numValues(); j++) {
				
				int c = Integer.parseInt(data.classAttribute().value(j));

				classProb[c] = _utils.calcPooledMean(classProb[c], numInstances, classProb2[c], numInstances2);
				
				for (int i = 0; i < data.numAttributes(); i++) {

					if (!data.attribute(i).equals(data.classAttribute())) {

						mu[i][c] = _utils.calcPooledMean(mu[i][c], numInstances, mu2[i][c], numInstances2);

						sd[i][c] = _utils.calcPooledSTD(sd[i][c], numInstances, sd2[i][c], numInstances2);

					}
				}

			}
			
			numInstances += numInstances2;

		}
	}

	public ClassifierSplitModel getLocalModel() {
		return m_localModel;
	}

	public ClassifierTree[] getSons() {
		return m_sons;
	}

	public boolean isLeaf() {
		return m_isLeaf;
	}

	public Instances getTrainingData() {
		return m_train;
	}

	/**
	 * Gets the next unique node ID.
	 * 
	 * @return the next unique node ID.
	 */
	protected static long nextID() {

		return PRINTED_NODES++;
	}

	/**
	 * Resets the unique node ID counter (e.g. between repeated separate print
	 * types)
	 */
	protected static void resetID() {

		PRINTED_NODES = 0;
	}

	/**
	 * Returns default capabilities of the classifier tree.
	 *
	 * @return the capabilities of this classifier tree
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = new Capabilities(this);
		result.enableAll();

		return result;
	}

	/**
	 * Constructor.
	 */
	public ClassifierTree(ModelSelection toSelectLocModel) {

		m_toSelectModel = toSelectLocModel;
	}

	/**
	 * Method for building a classifier tree.
	 * 
	 * @param data the data to build the tree from
	 * @throws Exception if something goes wrong
	 */
	public void buildClassifier(Instances data) throws Exception {

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();
		buildTree(data, false);
	}

	/**
	 * Builds the tree structure.
	 * 
	 * @param data     the data for which the tree structure is to be generated.
	 * @param keepData is training data to be kept?
	 * @throws Exception if something goes wrong
	 */
	public void buildTree(Instances data, boolean keepData) throws Exception {

		if (keepData) {
			m_train = data;
		}
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		m_localModel = m_toSelectModel.selectModel(data);

		Instances[] localInstances;

		this.setParameters(data);

		if (m_localModel.numSubsets() > 1 && this.depth < 0) {
			localInstances = m_localModel.split(data);
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
			for (int i = 0; i < m_sons.length; i++) {

				m_sons[i] = getNewTree(localInstances[i], depth + 1);
//				System.out.println("jjj ba moohaye soorakh");
//				if ( m_sons[i].isLeaf() == true )
//					m_sons[i].setParameters(localInstances[i]);

				if (m_sons[i].m_isLeaf == true && m_sons[i].mu == null) {
					System.out.println("sfdafsdfsdafafshiit");
					System.exit(0);
				}
				localInstances[i] = null;
			}
		} else {

//			this.setParameters(data);

			if (this.mu == null || this.classProb == null || this.sd == null) {
				System.out.println("mohammad e ali poor");

//				this.setParameters(data);

				System.exit(0);
			}
			m_isLeaf = true;
			if (Utils.eq(data.sumOfWeights(), 0)) {
				m_isEmpty = true;
			}
			data = null;
		}
	}

	/**
	 * Builds the tree structure with hold out set
	 * 
	 * @param train    the data for which the tree structure is to be generated.
	 * @param test     the test data for potential pruning
	 * @param keepData is training Data to be kept?
	 * @throws Exception if something goes wrong
	 */
	public void buildTree(Instances train, Instances test, boolean keepData) throws Exception {

		Instances[] localTrain, localTest;
		int i;

		if (keepData) {
			m_train = train;
		}
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		m_localModel = m_toSelectModel.selectModel(train, test);
		m_test = new Distribution(test, m_localModel);
		if (m_localModel.numSubsets() > 1) {
			localTrain = m_localModel.split(train);
			localTest = m_localModel.split(test);
			train = null;
			test = null;
			m_sons = new ClassifierTree[m_localModel.numSubsets()];
			for (i = 0; i < m_sons.length; i++) {
				m_sons[i] = getNewTree(localTrain[i], localTest[i]);
				localTrain[i] = null;
				localTest[i] = null;
			}
		} else {
			System.out.println("omg");
			System.exit(0);
			m_isLeaf = true;
			if (Utils.eq(train.sumOfWeights(), 0)) {
				m_isEmpty = true;
			}
			train = null;
			test = null;
		}
	}

	/**
	 * Classifies an instance.
	 * 
	 * @param instance the instance to classify
	 * @return the classification
	 * @throws Exception if something goes wrong
	 */
	public double classifyInstance(Instance instance) throws Exception {

		double maxProb = -1;
		double currentProb;
		int maxIndex = 0;
		int j;

		for (j = 0; j < instance.numClasses(); j++) {
			currentProb = getProbs(j, instance, 1);
			if (Utils.gr(currentProb, maxProb)) {
				maxIndex = j;
				maxProb = currentProb;
			}
		}

		return maxIndex;
	}

	/**
	 * Cleanup in order to save memory.
	 * 
	 * @param justHeaderInfo
	 */
	public final void cleanup(Instances justHeaderInfo) {

		m_train = justHeaderInfo;
		m_test = null;
		if (!m_isLeaf) {
			for (ClassifierTree m_son : m_sons) {
				boolean pre = m_son.isLeaf();
				m_son.cleanup(justHeaderInfo);
				if (pre != m_son.isLeaf()) {
					System.out.println("I found the bug");
					System.exit(0);
				}
			}
		}
	}

	/**
	 * Returns class probabilities for a weighted instance.
	 * 
	 * @param instance   the instance to get the distribution for
	 * @param useLaplace whether to use laplace or not
	 * @return the distribution
	 * @throws Exception if something goes wrong
	 */
	public final double[] distributionForInstance(Instance instance, boolean useLaplace) throws Exception {

		double[] doubles = new double[instance.numClasses()];

		for (int i = 0; i < doubles.length; i++) {
			if (!useLaplace) {
				doubles[i] = getProbs(i, instance, 1);
			} else {
				doubles[i] = getProbsLaplace(i, instance, 1);
			}
		}

		return doubles;
	}

	/**
	 * Assigns a uniqe id to every node in the tree.
	 * 
	 * @param lastID the last ID that was assign
	 * @return the new current ID
	 */
	public int assignIDs(int lastID) {

		int currLastID = lastID + 1;

		m_id = currLastID;
		if (m_sons != null) {
			for (ClassifierTree m_son : m_sons) {
				currLastID = m_son.assignIDs(currLastID);
			}
		}
		return currLastID;
	}

	/**
	 * Returns the type of graph this classifier represents.
	 * 
	 * @return Drawable.TREE
	 */
	@Override
	public int graphType() {
		return Drawable.TREE;
	}

	/**
	 * Returns graph describing the tree.
	 * 
	 * @throws Exception if something goes wrong
	 * @return the tree as graph
	 */
	@Override
	public String graph() throws Exception {

		StringBuffer text = new StringBuffer();

		assignIDs(-1);
		text.append("digraph J48Tree {\n");
		if (m_isLeaf) {
			text.append("N" + m_id + " [label=\"" + Utils.backQuoteChars(m_localModel.dumpLabel(0, m_train)) + "\" "
					+ "shape=box style=filled ");
			if (m_train != null && m_train.numInstances() > 0) {
				text.append("data =\n" + m_train + "\n");
				text.append(",\n");

			}
			text.append("]\n");
		} else {
			text.append("N" + m_id + " [label=\"" + Utils.backQuoteChars(m_localModel.leftSide(m_train)) + "\" ");
			if (m_train != null && m_train.numInstances() > 0) {
				text.append("data =\n" + m_train + "\n");
				text.append(",\n");
			}
			text.append("]\n");
			graphTree(text);
		}

		return text.toString() + "}\n";
	}

	/**
	 * Returns tree in prefix order.
	 * 
	 * @throws Exception if something goes wrong
	 * @return the prefix order
	 */
	public String prefix() throws Exception {

		StringBuffer text;

		text = new StringBuffer();
		if (m_isLeaf) {
			text.append("[" + m_localModel.dumpLabel(0, m_train) + "]");
		} else {
			prefixTree(text);
		}

		return text.toString();
	}

	/**
	 * Returns source code for the tree as an if-then statement. The class is
	 * assigned to variable "p", and assumes the tested instance is named "i". The
	 * results are returned as two stringbuffers: a section of code for assignment
	 * of the class, and a section of code containing support code (eg: other
	 * support methods).
	 * 
	 * @param className the classname that this static classifier has
	 * @return an array containing two stringbuffers, the first string containing
	 *         assignment code, and the second containing source for support code.
	 * @throws Exception if something goes wrong
	 */
	public StringBuffer[] toSource(String className) throws Exception {

		StringBuffer[] result = new StringBuffer[2];
		if (m_isLeaf) {
			result[0] = new StringBuffer("    p = " + m_localModel.distribution().maxClass(0) + ";\n");
			result[1] = new StringBuffer("");
		} else {
			StringBuffer text = new StringBuffer();
			StringBuffer atEnd = new StringBuffer();

			long printID = ClassifierTree.nextID();

			text.append("  static double N").append(Integer.toHexString(m_localModel.hashCode()) + printID)
					.append("(Object []i) {\n").append("    double p = Double.NaN;\n");

			text.append("    if (").append(m_localModel.sourceExpression(-1, m_train)).append(") {\n");
			text.append("      p = ").append(m_localModel.distribution().maxClass(0)).append(";\n");
			text.append("    } ");
			for (int i = 0; i < m_sons.length; i++) {
				text.append("else if (" + m_localModel.sourceExpression(i, m_train) + ") {\n");
				if (m_sons[i].m_isLeaf) {
					text.append("      p = " + m_localModel.distribution().maxClass(i) + ";\n");
				} else {
					StringBuffer[] sub = m_sons[i].toSource(className);
					text.append(sub[0]);
					atEnd.append(sub[1]);
				}
				text.append("    } ");
				if (i == m_sons.length - 1) {
					text.append('\n');
				}
			}

			text.append("    return p;\n  }\n");

			result[0] = new StringBuffer("    p = " + className + ".N");
			result[0].append(Integer.toHexString(m_localModel.hashCode()) + printID).append("(i);\n");
			result[1] = text.append(atEnd);
		}
		return result;
	}

	/**
	 * Returns number of leaves in tree structure.
	 * 
	 * @return the number of leaves
	 */
	public int numLeaves() {

		int num = 0;
		int i;

		if (m_isLeaf) {
			return 1;
		} else {
			for (i = 0; i < m_sons.length; i++) {
				num = num + m_sons[i].numLeaves();
			}
		}

		return num;
	}

	/**
	 * Returns number of nodes in tree structure.
	 * 
	 * @return the number of nodes
	 */
	public int numNodes() {

		int no = 1;
		int i;

		if (!m_isLeaf) {
			for (i = 0; i < m_sons.length; i++) {
				no = no + m_sons[i].numNodes();
			}
		}

		return no;
	}

	/**
	 * Prints tree structure.
	 * 
	 * @return the tree structure
	 */
	@Override
	public String toString() {

		try {
			StringBuffer text = new StringBuffer();

			if (m_isLeaf) {
				text.append(": ");
				text.append(m_localModel.dumpLabel(0, m_train));
			} else {
				dumpTree(0, text);
			}
			text.append("\n\nNumber of Leaves  : \t" + numLeaves() + "\n");
			text.append("\nSize of the tree : \t" + numNodes() + "\n");

			return text.toString();
		} catch (Exception e) {
			return "Can't print classification tree.";
		}
	}

	/**
	 * Returns a newly created tree.
	 * 
	 * @param data the training data
	 * @return the generated tree
	 * @throws Exception if something goes wrong
	 */
	protected ClassifierTree getNewTree(Instances data, int depth) throws Exception {

		ClassifierTree newTree = new ClassifierTree(m_toSelectModel);
		this.depth = depth;

		newTree.buildTree(data, false);

		return newTree;
	}

	/**
	 * Returns a newly created tree.
	 * 
	 * @param train the training data
	 * @param test  the pruning data.
	 * @return the generated tree
	 * @throws Exception if something goes wrong
	 */
	protected ClassifierTree getNewTree(Instances train, Instances test) throws Exception {

		ClassifierTree newTree = new ClassifierTree(m_toSelectModel);
		newTree.buildTree(train, test, false);

		return newTree;
	}

	/**
	 * Help method for printing tree structure.
	 * 
	 * @param depth the current depth
	 * @param text  for outputting the structure
	 * @throws Exception if something goes wrong
	 */
	private void dumpTree(int depth, StringBuffer text) throws Exception {

		int i, j;

		for (i = 0; i < m_sons.length; i++) {
			text.append("\n");
			;
			for (j = 0; j < depth; j++) {
				text.append("|   ");
			}
			text.append(m_localModel.leftSide(m_train));
			text.append(m_localModel.rightSide(i, m_train));
			if (m_sons[i].m_isLeaf) {
				text.append(": ");
				text.append(m_localModel.dumpLabel(i, m_train));
			} else {
				m_sons[i].dumpTree(depth + 1, text);
			}
		}
	}

	/**
	 * Help method for printing tree structure as a graph.
	 * 
	 * @param text for outputting the tree
	 * @throws Exception if something goes wrong
	 */
	private void graphTree(StringBuffer text) throws Exception {

		for (int i = 0; i < m_sons.length; i++) {
			text.append("N" + m_id + "->" + "N" + m_sons[i].m_id + " [label=\""
					+ Utils.backQuoteChars(m_localModel.rightSide(i, m_train).trim()) + "\"]\n");
			if (m_sons[i].m_isLeaf) {
				text.append("N" + m_sons[i].m_id + " [label=\""
						+ Utils.backQuoteChars(m_localModel.dumpLabel(i, m_train)) + "\" " + "shape=box style=filled ");
				if (m_train != null && m_train.numInstances() > 0) {
					text.append("data =\n" + m_sons[i].m_train + "\n");
					text.append(",\n");
				}
				text.append("]\n");
			} else {
				text.append("N" + m_sons[i].m_id + " [label=\""
						+ Utils.backQuoteChars(m_sons[i].m_localModel.leftSide(m_train)) + "\" ");
				if (m_train != null && m_train.numInstances() > 0) {
					text.append("data =\n" + m_sons[i].m_train + "\n");
					text.append(",\n");
				}
				text.append("]\n");
				m_sons[i].graphTree(text);
			}
		}
	}

	/**
	 * Prints the tree in prefix form
	 * 
	 * @param text the buffer to output the prefix form to
	 * @throws Exception if something goes wrong
	 */
	private void prefixTree(StringBuffer text) throws Exception {

		text.append("[");
		text.append(m_localModel.leftSide(m_train) + ":");
		for (int i = 0; i < m_sons.length; i++) {
			if (i > 0) {
				text.append(",\n");
			}
			text.append(m_localModel.rightSide(i, m_train));
		}
		for (int i = 0; i < m_sons.length; i++) {
			if (m_sons[i].m_isLeaf) {
				text.append("[");
				text.append(m_localModel.dumpLabel(i, m_train));
				text.append("]");
			} else {
				m_sons[i].prefixTree(text);
			}
		}
		text.append("]");
	}

	/**
	 * Help method for computing class probabilities of a given instance.
	 * 
	 * @param classIndex the class index
	 * @param instance   the instance to compute the probabilities for
	 * @param weight     the weight to use
	 * @return the laplace probs
	 * @throws Exception if something goes wrong
	 */
	private double getProbsLaplace(int classIndex, Instance instance, double weight) throws Exception {

		double prob = 0;

		if (m_isLeaf) {
			return weight * localModel().classProbLaplace(classIndex, instance, -1);
		} else {
			int treeIndex = localModel().whichSubset(instance);
			if (treeIndex == -1) {
				double[] weights = localModel().weights(instance);
				for (int i = 0; i < m_sons.length; i++) {
					if (!son(i).m_isEmpty) {
						prob += son(i).getProbsLaplace(classIndex, instance, weights[i] * weight);
					}
				}
				return prob;
			} else {
				if (son(treeIndex).m_isEmpty) {
					return weight * localModel().classProbLaplace(classIndex, instance, treeIndex);
				} else {
					return son(treeIndex).getProbsLaplace(classIndex, instance, weight);
				}
			}
		}
	}

	/**
	 * Help method for computing class probabilities of a given instance.
	 * 
	 * @param classIndex the class index
	 * @param instance   the instance to compute the probabilities for
	 * @param weight     the weight to use
	 * @return the probs
	 * @throws Exception if something goes wrong
	 */
	private double getProbs(int classIndex, Instance instance, double weight) throws Exception {

		double prob = 0;
		System.out.println("wrong place");
//		System.exit(0);
		if (m_isLeaf) {
			return weight * localModel().classProb(classIndex, instance, -1);
		} else {
			int treeIndex = localModel().whichSubset(instance);
			if (treeIndex == -1) {
				double[] weights = localModel().weights(instance);
				for (int i = 0; i < m_sons.length; i++) {
					if (!son(i).m_isEmpty) {
						prob += son(i).getProbs(classIndex, instance, weights[i] * weight);
					}
				}
				return prob;
			} else {
				if (son(treeIndex).m_isEmpty) {
					return weight * localModel().classProb(classIndex, instance, treeIndex);
				} else {
					return son(treeIndex).getProbs(classIndex, instance, weight);
				}
			}
		}
	}

	/**
	 * Method just exists to make program easier to read.
	 */
	private ClassifierSplitModel localModel() {

		return m_localModel;
	}

	/**
	 * Method just exists to make program easier to read.
	 */
	private ClassifierTree son(int index) {

		return m_sons[index];
	}

	/**
	 * Computes a list that indicates node membership
	 */
	public double[] getMembershipValues(Instance instance) throws Exception {

		// Set up array for membership values
		double[] a = new double[numNodes()];

		// Initialize queues
		Queue<Double> queueOfWeights = new LinkedList<Double>();
		Queue<ClassifierTree> queueOfNodes = new LinkedList<ClassifierTree>();
		queueOfWeights.add(instance.weight());
		queueOfNodes.add(this);
		int index = 0;

		// While the queue is not empty
		while (!queueOfNodes.isEmpty()) {

			a[index++] = queueOfWeights.poll();
			ClassifierTree node = queueOfNodes.poll();

			// Is node a leaf?
			if (node.m_isLeaf) {
				continue;
			}

			// Which subset?
			int treeIndex = node.localModel().whichSubset(instance);

			// Space for weight distribution
			double[] weights = new double[node.m_sons.length];

			// Check for missing value
			if (treeIndex == -1) {
				weights = node.localModel().weights(instance);
			} else {
				weights[treeIndex] = 1.0;
			}
			for (int i = 0; i < node.m_sons.length; i++) {
				queueOfNodes.add(node.son(i));
				queueOfWeights.add(a[index - 1] * weights[i]);
			}
		}
		return a;
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 14534 $");
	}

}

