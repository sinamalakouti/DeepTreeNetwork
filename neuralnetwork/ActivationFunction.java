package neuralnetwork;



import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.complex.Complex;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.NDArrayUtil;

import tree.DecisionTree;
import tree.MyREPTree;
import utils.FourierUtils;
import utils._utils;
import weka.classifiers.trees.ht.HNode;
import weka.core.Instances;
import weka.core.WekaException;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class ActivationFunction extends BaseActivationFunction  {
/* this is for hoeffding tree: */
	private DecisionTree activationModel; 
//	this is for REPTree:
//	private MyREPTree activationModel;
	private Instances trainInstances;
	private Instances testInstances;
	private boolean isFirstLayer;
	
	public  ArrayList<Integer> attCardinality = new ArrayList<>();
	public  int completeInstanceSpace = 0;
	public  HashMap<HNode, Double> Fxk = new HashMap<>();
	public  HashMap<ArrayList<String> , Complex > weights ;
//	public  List<ArrayList<Double>> cutpoints = new ArrayList<>();
		
	
	
	

	public ActivationFunction(Instances trainInstances, Instances testInstances, boolean isFirstLayer) {
		this.trainInstances = trainInstances;
		this.testInstances = testInstances;
		this.isFirstLayer = isFirstLayer;
	}

	@Override
	public INDArray getActivation(INDArray in, boolean arg1) {

		//		INDArray	in2 = in.dup().getColumns(0);
		//		in2  = org.nd4j.linalg.ops.transforms.Transforms.sigmoid(in2, false);

		//		Nd4j.getExecutioner().execAndReturn(new Sigmoid(in));
		//		in.setData( in.dup().data()  );
//		in  = org.nd4j.linalg.ops.transforms.Transforms.sigmoid(in, false);
		
		System.out.println("lets do get activations");
		
		double[][] result = null;
		try {
			result = this.getActivationFunResult(in, arg1);
		} catch (Exception e) {
			e.printStackTrace();
		}
		in = Nd4j.create(result);
		return in;
	}



	@Override
	public org.nd4j.linalg.primitives.Pair<INDArray, INDArray> backprop(INDArray arg0, INDArray arg1) {
		double[][] result = null;
		
		if (activationModel.m_root.isLeaf() == false) {
			
			_utils.setAttCardinality(activationModel.cutpoints, this);
			if ( attCardinality.size() == 3 &&! this.attCardinality.contains(0) ) {
				try {
					FourierUtils.setFourierSeriesWeights(activationModel.m_root,this);
				} catch (Exception e) {
					e.printStackTrace();
				}

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
			
			}

		}
		try {
			result = this.getActivationFunResult(arg0, true);
		} catch (Exception e) {
			e.printStackTrace();
		}
		INDArray dLdz = Nd4j.create(result);
		dLdz.muli(arg1);

		return new org.nd4j.linalg.primitives.Pair<>(dLdz, null);	}
	
	
	
	

	private double [][] getActivationFunResult (INDArray arg0 , boolean arg1) throws Exception{

		Instances instances = null;
		try {
			INDArray label = null;
			int[] arr = null;
			if (arg1 == false) {
				arr = _utils.getLabels(testInstances);
			} else
				arr = _utils.getLabels(trainInstances);

			label = NDArrayUtil.toNDArray(arr);
			label = label.transpose();
			INDArray dataset = Nd4j.concat(1, arg0, label);
			instances = _utils.ndArrayToInstances(dataset);
			instances.setClassIndex(instances.numAttributes() - 1);
			NumericToNominal convert = new NumericToNominal();
			String[] options = new String[2];
			options[0] = "-R";
			options[1] = "" + (instances.classIndex() + 1); // range of variables to make numeric
			convert.setOptions(options);
			convert.setInputFormat(instances);
			instances = weka.filters.Filter.useFilter(instances, convert);




			//			System.out.println(instances.size());
		} catch (WekaException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}


// 			this is for hoeffding tree : 
		if (activationModel == null) {
			try {
				activationModel = new DecisionTree(3, instances);
			} catch (Exception e) {
				e.printStackTrace();
			}


		}
		else {


			try {
				activationModel.updateClassifier(instances,this);
			} catch (WekaException e) {
				e.printStackTrace();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		
//		this is for REPTree:
		
//		this.activationModel = new MyREPTree(5, instances);


		double[][] result = new double[instances.size()][3];
		for (int i = 0; i < instances.size(); i++) {
			try {
				double value = activationModel.classifyInstance(instances.get(i));
			} catch (Exception e) {
				e.printStackTrace();
			}
			double[] percentage = null;

			try {
				percentage = activationModel.distributionForInstance(instances.get(i));


			} catch (Exception e) {
				e.printStackTrace();
			}
			result[i] = percentage;

		}
		return result;
	}


}



