package neuralnetwork;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.NDArrayUtil;

import utils._utils;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.WekaException;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class BayesTreeActivationFunction extends BaseActivationFunction{

	
	private J48 activationModel; 
//	this is for REPTree:
//	private MyREPTree activationModel;
	private Instances trainInstances;
	private Instances testInstances;
	
	
	public BayesTreeActivationFunction(Instances trainInstances, Instances testInstances, boolean isFirstLayer) {
		this.trainInstances = trainInstances;
		this.testInstances = testInstances;
	}
	
	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
//		TODO : check the correction of reuslt.muli(epsilon)
		Instances trainInstaces = createProperDataset(in, true);
		double [] result = new double [trainInstaces.size()];
		
		
		for ( int i = 0 ; i < trainInstaces.size() ; i++) {
			try {
				double[] prediciton = activationModel.predicateDerivative(trainInstaces.get(i));
				double interval = 1d / trainInstaces.numClasses();
				double res = prediciton[0]/ 1d * interval  + prediciton[1] * interval;
				result[i] = res;
				if ( res > 1 ) {
//					System.out.println("probability greater than 1 ahha");
//					System.exit(0);
				}
					
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
		}
		
//		result = result.muli(epsilon);

		return new Pair<>(Nd4j.create(result), null);
	}

	@Override
	public INDArray getActivation(INDArray in, boolean training) {
		// in  =  inputData .* weights
		Instances trainInstaces = createProperDataset(in, training);
		activationModel = new J48();
		double [] result = new double [trainInstaces.size()];
		try {
			activationModel.buildClassifier(trainInstaces);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		for ( int i = 0 ; i < trainInstaces.size() ; i++) {
			try {
				double[] prediciton = activationModel.predicate(trainInstaces.get(i));
				double interval = 1d / trainInstaces.numClasses();
				double res = prediciton[0]/ 1d * 0.33  + prediciton[1] * interval;
				result[i] = res;
				if ( res > 1 ) {
					System.out.println("probability greater than 1 ahha");
					System.exit(0);
				}
					
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
		}
		
		return Nd4j.create(result);
	}

	
	private Instances createProperDataset(INDArray in, boolean training) {
		

		Instances instances = null;
		try {
			INDArray label = null;
			int[] arr = null;
			if (training == false) {
				arr = _utils.getLabels(testInstances);
			} else
				arr = _utils.getLabels(trainInstances);

			label = NDArrayUtil.toNDArray(arr);
			label = label.transpose();
			INDArray dataset = Nd4j.concat(1, in, label);
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
		return instances;

	}
}
