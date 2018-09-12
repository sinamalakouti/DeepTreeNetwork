package neuralnetwork;

import java.util.Enumeration;
import java.util.Iterator;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.NDArrayUtil;

import play.api.libs.iteratee.Enumerator;
import utils.Constants;
import utils._utils;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class BayesTreeActivationFunction extends BaseActivationFunction {

	private J48 activationModel;

//	public BayesTreeActivationFunction(Instances trainInstances, Instances testInstances, boolean isFirstLayer) {
////		testInstancesLabel = NDArrayUtil.toNDArray(_utils.getLabels(testInstances)).transpose();
////		trainInstancesLabel = NDArrayUtil.toNDArray(_utils.getLabels(trainInstances)).transpose();
//	}

	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
//		TODO : check the correction of reuslt.muli(epsilon)
		Instances trainInstaces = createProperDataset(in, true);
		double[] result = new double[trainInstaces.size()];

		Enumeration<Instance> it = trainInstaces.enumerateInstances();
 
		int i = 0; 
		while (it.hasMoreElements()) {
			try {
				double[] prediciton = activationModel.predicateDerivative(it.nextElement());
				double interval = 1d / trainInstaces.numClasses();
				double res = prediciton[0] / 1d * interval + prediciton[1];
				if (res > 2) {
//					System.out.println("okh okh");
				
//					prediciton = activationModel.predicateDerivative(is);

				}
				result[i] = res;
				if (res > 1) {
//					System.out.println("probability greater than 1 ahha");
//					System.exit(0);
				}

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			i++;
		}

//		result = result.muli(epsilon);

		return new Pair<>(Nd4j.create(result), null);
	}

	@Override
	public INDArray getActivation(INDArray in, boolean training) {
		// in = inputData .* weights
		Instances trainInstaces = createProperDataset(in, training);
		activationModel = new J48();
		double[] result = new double[trainInstaces.size()];

		try {
			activationModel.buildClassifier(trainInstaces);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		Iterator<Instance> it = trainInstaces.iterator();
		int i = 0;
		while (it.hasNext()) {

			double[] prediciton;
			try {
				prediciton = activationModel.predicate(it.next());
				double interval = 1d / trainInstaces.numClasses();
				double res = prediciton[0] / 1d * interval + prediciton[1] * interval;
				result[i] = res;
				if (res > 1) {
					System.out.println("probability greater than 1 ahha");
					System.exit(0);
				}

			} catch (Exception e) {
				e.printStackTrace();
			}

			i++;

		}


		return Nd4j.create(result);
	}

	private Instances createProperDataset(INDArray in, boolean training) {
		Instances instances = null;

		try {
			INDArray label = null;
			if (training == false)
				label = Constants.testInstancesLabel;
			else
				label = Constants.trainInstancesLabel;

			INDArray dataset = Nd4j.concat(1, in, label);
			instances = _utils.ndArrayToInstances(dataset);
			instances.setClassIndex(instances.numAttributes() - 1);

			if (!instances.classAttribute().isNominal()) {
				NumericToNominal convert = new NumericToNominal();
				String[] options = new String[2];
				options[0] = "-R";
				options[1] = "" + (instances.classIndex() + 1); // range of variables to make numeric
				convert.setOptions(options);
				convert.setInputFormat(instances);
				instances = weka.filters.Filter.useFilter(instances, convert);
			}
			// System.out.println(instances.size());
		} catch (WekaException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return instances;

	}
}
