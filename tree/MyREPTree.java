package tree;

import java.awt.BorderLayout;

import utils.FourierUtils;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class MyREPTree extends REPTree{
	
	
	public MyREPTree (int maximumDepth, Instances data) throws Exception {
			
		String [] options = new String[2];
		options[0] =  "-L" ;
		options[1] =  "" + maximumDepth;
		super.setOptions(options);
		
		super.buildClassifier(data);
		
		final javax.swing.JFrame jf = 
			       new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
			     jf.setSize(500,400);
			     jf.getContentPane().setLayout(new BorderLayout());
			     TreeVisualizer tv = new TreeVisualizer(null,
			         this.graph(),
			         new PlaceNode2());
			     jf.getContentPane().add(tv, BorderLayout.CENTER);
			     jf.addWindowListener(new java.awt.event.WindowAdapter() {
			       public void windowClosing(java.awt.event.WindowEvent e) {
			         jf.dispose();
			       }
			     });

			     jf.setVisible(true);
			     tv.fitToScreen();
//	FourierUtils.setFourierSeriesWeights(super.));

	}
	
	public int getMaxDepth () {
		return super.m_MaxDepth;
		
	}
	
	
	

	
	
	
	

}
