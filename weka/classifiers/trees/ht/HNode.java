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
 *    HNode.java
 *    Copyright (C) 2013 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.trees.ht;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;

import utils.Constants;
import utils._utils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * Abstract base class for nodes in a Hoeffding tree
 * 
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @revision $Revision: 9707 $
 */
public abstract class HNode implements Serializable {
	
 /**
  * 
  * added by sina :D  
  * 
  * 
  * 
  */
	
	
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
	
	
	 public abstract double[] getDistribution_derivative(Instance inst, Attribute classAtt) throws Exception;

	 
	 
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
	
	
  /**
   * For serialization
   */
  private static final long serialVersionUID = 197233928177240264L;

  /** Class distribution at this node */
  public Map<String, WeightMass> m_classDistribution = new LinkedHashMap<String, WeightMass>();

  /** Holds the leaf number (if this is a leaf) */
  protected int m_leafNum;

  /** Holds the node number (for graphing purposes) */
  protected int m_nodeNum;

  /**
   * Construct a new HNode
   */
  public HNode() {
  }

  /**
   * Construct a new HNode with the supplied class distribution
   * 
   * @param classDistrib
   */
  public HNode(Map<String, WeightMass> classDistrib) {
    m_classDistribution = classDistrib;
  }

  /**
   * Returns true if this is a leaf
   * 
   * @return
   */
  public boolean isLeaf() {
    return true;
  }

  /**
   * The size of the class distribution
   * 
   * @return the number of entries in the class distribution
   */
  public int numEntriesInClassDistribution() {
    return m_classDistribution.size();
  }

  /**
   * Returns true if the class distribution is pure
   * 
   * @return true if the class distribution is pure
   */
  public boolean classDistributionIsPure() {
    int count = 0;
    for (Map.Entry<String, WeightMass> el : m_classDistribution.entrySet()) {
      if (el.getValue().m_weight > 0) {
        count++;

        if (count > 1) {
          break;
        }
      }
    }

    return (count < 2);
  }

  /**
   * Update the class frequency distribution with the supplied instance
   * 
   * @param inst the instance to update with
   */
  public void updateDistribution(Instance inst) {
    if (inst.classIsMissing()) {
      return;
    }
    String classVal = inst.stringValue(inst.classAttribute());

    WeightMass m = m_classDistribution.get(classVal);
    if (m == null) {
      m = new WeightMass();
      m.m_weight = 1.0;

      m_classDistribution.put(classVal, m);
    }
    m.m_weight += inst.weight();
  }

  /**
   * Return a class probability distribution computed from the frequency counts
   * at this node
   * 
   * @param inst the instance to get a prediction for
   * @param classAtt the class attribute
   * @return a class probability distribution
   * @throws Exception if a problem occurs
   */
  public double[] getDistribution(Instance inst, Attribute classAtt)
      throws Exception {
    double[] dist = new double[classAtt.numValues()];

    for (int i = 0; i < classAtt.numValues(); i++) {
      WeightMass w = m_classDistribution.get(classAtt.value(i));
      if (w != null) {
        dist[i] = w.m_weight;
      } else {
        dist[i] = 1.0;
      }
    }

    Utils.normalize(dist);
    return dist;
  }

  public int installNodeNums(int nodeNum) {
    nodeNum++;
    m_nodeNum = nodeNum;

    return nodeNum;
  }

  protected int dumpTree(int depth, int leafCount, StringBuffer buff) {

    double max = -1;
    String classVal = "";
    for (Map.Entry<String, WeightMass> e : m_classDistribution.entrySet()) {
      if (e.getValue().m_weight > max) {
        max = e.getValue().m_weight;
        classVal = e.getKey();
      }
    }
    buff.append(classVal + " (" + String.format("%-9.3f", max).trim() + ")");
    leafCount++;
    m_leafNum = leafCount;

    return leafCount;
  }

  protected void printLeafModels(StringBuffer buff) {
  }

  public void graphTree(StringBuffer text) {

    double max = -1;
    String classVal = "";
    for (Map.Entry<String, WeightMass> e : m_classDistribution.entrySet()) {
      if (e.getValue().m_weight > max) {
        max = e.getValue().m_weight;
        classVal = e.getKey();
      }
    }

    text.append("N" + m_nodeNum + " [label=\"" + classVal + " ("
        + String.format("%-9.3f", max).trim() + ")\" shape=box style=filled]\n");
  }

  /**
   * Print a textual description of the tree
   * 
   * @param printLeaf true if leaf models (NB, NB adaptive) should be output
   * @return a textual description of the tree
   */
  public String toString(boolean printLeaf) {

    installNodeNums(0);

    StringBuffer buff = new StringBuffer();

    dumpTree(0, 0, buff);

    if (printLeaf) {
      buff.append("\n\n");
      printLeafModels(buff);
    }

    return buff.toString();
  }

  /**
   * Return the total weight of instances seen at this node
   * 
   * @return the total weight of instances seen at this node
   */
  public double totalWeight() {
    double tw = 0;

    for (Map.Entry<String, WeightMass> e : m_classDistribution.entrySet()) {
      tw += e.getValue().m_weight;
    }

    return tw;
  }

  /**
   * Return the leaf that the supplied instance ends up at
   * 
   * @param inst the instance to find the leaf for
   * @param parent the parent node
   * @param parentBranch the parent branch
   * @return the leaf that the supplied instance ends up at
   */
  public LeafNode leafForInstance(Instance inst, SplitNode parent,
	      String parentBranch) {
		  return new LeafNode(this, parent, parentBranch);
	  }

  /**
   * Update the node with the supplied instance
   * 
   * @param inst the instance to update with
   * @throws Exception if a problem occurs
   */
  public abstract void updateNode(Instance inst) throws Exception;
}
