package controller;

import java.io.IOException;

import utils._utils;

//import neuralnetwork.HoeffdingTree.HoeffdingTreeActivationFunction;
//import utils.Constants;
//import weka.classifiers.trees.HoeffdingTree;
//import weka.gui.treevisualizer.PlaceNode2;
//import weka.gui.treevisualizer.TreeVisualizer;

public class Test {






	//	public  ArrayList<HashMap<Integer, HoeffdingTree>>  model1 = new ArrayList<>();
	//	public  ArrayList<HashMap<Integer, HoeffdingTree>>  model2 = new ArrayList<>();


	//	public static void main(String[] args) throws Exception {
	//		
	//		Test test = new Test();
	//		final int numInputs = 9;
	//		int outputNum = 3;
	//		Constants.numberOfLayers = 8;
	//		Constants.numberOfNeurons = 5;
	//		int neuron_feature_ratio = 2;
	//		Constants.batchSize = 100;
	//		Constants.avgHFDepth = new double[Constants.numberOfLayers];
	//		double numberOfExamples = 10000d;
	//		double numberTrainExamples = 7000d;
	//		Constants.numBatches = (int) ( (numberTrainExamples) / Constants.batchSize); 
	//		Constants.numClasses = 3;
	//
	//		
	//		for (int l =0 ; l < Constants.numberOfLayers ; l++){
	//			HashMap<Integer, HoeffdingTree> hfs2 =new HashMap<>();	
	//			for ( int jj =0 ; jj < Constants.numberOfNeurons ;jj ++){
	//
	//				FileInputStream file = null;
	//				ObjectInputStream in = null;
	//				try {
	//					file = new FileInputStream("hf_Activation_"+ l + "_" + jj);
	//				} catch (FileNotFoundException e1) {
	//					// TODO Auto-generated catch block
	//					e1.printStackTrace();
	//				} 
	//				try {
	//					in = new ObjectInputStream(file);
	//				} catch (IOException e1) {
	//					// TODO Auto-generated catch block
	//					e1.printStackTrace();
	//				} 
	//				HoeffdingTreeActivationFunction object1 = null;
	//				// Method for deserialization of object 
	//				try {
	//					object1 = (HoeffdingTreeActivationFunction)in.readObject();
	//				} catch (ClassNotFoundException | IOException e) {
	//					
	//					e.printStackTrace();
	//				} 
	//
	//				hfs2.put(jj, object1.getActivationModel());
	//
	//			}
	//
	//			test.model1.add(hfs2);
	//
	//
	//
	//
	//		}
	////
	//		
	//		
	//
	//		for (int l =0 ; l < Constants.numberOfLayers ; l++){
	//			HashMap<Integer, HoeffdingTree> hfs2 =new HashMap<>();	
	//			for ( int jj =0 ; jj < Constants.numberOfNeurons ;jj ++){
	//
	//				FileInputStream file = null;
	//				ObjectInputStream in = null;
	//				try {
	//					file = new FileInputStream("trees2_"+ l + "_" + jj);
	//				} catch (FileNotFoundException e1) {
	//					// TODO Auto-generated catch block
	//					e1.printStackTrace();
	//				} 
	//				try {
	//					in = new ObjectInputStream(file);
	//				} catch (IOException e1) {
	//					// TODO Auto-generated catch block
	//					e1.printStackTrace();
	//				} 
	//				HoeffdingTree object1 = null;
	//				// Method for deserialization of object 
	//				try {
	//					object1 = (HoeffdingTree)in.readObject();
	//				} catch (ClassNotFoundException | IOException e) {
	//					
	//					e.printStackTrace();
	//				} 
	//
	//				hfs2.put(jj, object1);
	//
	//			}
	//
	//			test.model2.add(hfs2);
	//
	//
	//
	//
	//		}
	//		
	//		System.out.println("heres");
	//		
	//
	//	     // display classifier
	//	     final javax.swing.JFrame jf = 
	//	       new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
	//	     jf.setSize(500,400);
	////	     jf.getContentPane().setLayout(new BorderLayout());
	//	     TreeVisualizer tv = new TreeVisualizer(null,
	//	         test.model2.get(0).get(2).graph(),
	//	         new PlaceNode2());
	//	     jf.getContentPane().add(tv, BorderLayout.CENTER);
	//	     jf.addWindowListener(new java.awt.event.WindowAdapter() {
	//	       public void windowClosing(java.awt.event.WindowEvent e) {
	//	         jf.dispose();
	//	       }
	//	     });
	//
	//	     jf.setVisible(true);
	//	     tv.fitToScreen();y
	//
	//	}


	public static void main(String[] args) {
		try{
			_utils.createGNUPLOT_ds("net_50n_40fr", "/Users/sina/Documents/JGU_Research/ComplexNeuronsProject/Experiments/DeCoDeML Workshop/16/", 0, 150);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
