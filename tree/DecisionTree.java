package tree;

import java.awt.BorderLayout;

import neuralnetwork.ActivationFunction;
import utils.Constants;
import utils.FourierUtils;
import utils._utils;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.ht.HNode;
import weka.classifiers.trees.ht.SplitNode;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;


public class DecisionTree extends HoeffdingTree {
	
	private int  depth;
	
	public DecisionTree(int depth, Instances trainSet) throws Exception {
		this.depth = 0;
		
		String [] options = new String[2];
		options[0] = "-S 1";
		options[1] = "-L 0";
		
		super.buildClassifier(trainSet);
	}
	
	
	public int getNumberOfNodes() {
		
		return this.m_decisionNodeCount;
	}
	public HNode getRoot ()
	{
		
		return m_root;
	}
	public int getDepth() {
		HoeffdingTree t = new HoeffdingTree();
	

		return depth;
	}

	public void setDepth(int depth) {
		this.depth = depth;
	}
	
	
	public void traverse(ClassifierTree s ) {
		
		
		
	}

	
	public void updateClassifier(Instances arg0 , ActivationFunction activation) throws Exception {
//		if (m_root.isLeaf() == false) {
			
			_utils.setAttCardinality(cutpoints, activation);
//			if ( activation.attCardinality.size() == 2 &&! activation.attCardinality.contains(0) ) {
				FourierUtils.setFourierSeriesWeights(m_root, activation);

//				final javax.swing.JFrame jf = 
//					       new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
//					     jf.setSize(500,400);
//					     jf.getContentPane().setLayout(new BorderLayout());
//					     TreeVisualizer tv = new TreeVisualizer(null,
//					         this.graph(),
//					         new PlaceNode2());
//					     jf.getContentPane().add(tv, BorderLayout.CENTER);
//					     jf.addWindowListener(new java.awt.event.WindowAdapter() {
//					       public void windowClosing(java.awt.event.WindowEvent e) {
//					         jf.dispose();
//					       }
//					     });
//
//					     jf.setVisible(true);
//					     tv.fitToScreen();
			
//			}

//		}
		if( this.depth < 2)
		{
			for ( int i =0 ; i < arg0.size() ;  i ++) {
				super.updateClassifier(arg0.get(i));
			}
		}
	}
	
	
	
	
	
	

}
