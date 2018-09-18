package neuralnetwork;

import java.util.Enumeration;
import java.util.Iterator;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
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
	int layernumber = 0;
	boolean isTraind = false;
	Boolean isOutputLayerActivation = false;
//	public BayesTreeActivationFunction(Instances trainInstances, Instances testInstances, boolean isFirstLayer) {
////		testInstancesLabel = NDArrayUtil.toNDArray(_utils.getLabels(testInstances)).transpose();
////		trainInstancesLabel = NDArrayUtil.toNDArray(_utils.getLabels(trainInstances)).transpose();
//	}

	public BayesTreeActivationFunction(int layerNUmber, boolean isOutpuLayerActivation) {
		this.layernumber = layerNUmber;
		this.isOutputLayerActivation = isOutpuLayerActivation;

	}

	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon1) {
//		TODO : check the correction of reuslt.muli(epsilon)

//		if ( isOutputLayerActivation == true)
//        assertShape(in, epsilon);

		Instances trainInstaces = createProperDataset(in, true);
		double[] result = new double[trainInstaces.size()];

		Enumeration<Instance> it = trainInstaces.enumerateInstances();

		int i = 0;
		double[] labelIndexes = new double[trainInstaces.size()];
		double maxDerivative = Double.NEGATIVE_INFINITY;
		double minDerivative = Double.POSITIVE_INFINITY;
		double interval = 1d / trainInstaces.numClasses();
		double[][] outputLayerResult = null;
		if (isOutputLayerActivation == true)
			outputLayerResult = new double[trainInstaces.size()][trainInstaces.numClasses()];

		while (it.hasMoreElements()) {

			try {
				double[] prediciton = activationModel.predicateDerivative(it.nextElement(), isOutputLayerActivation);
				if (isOutputLayerActivation == false) {
					labelIndexes[i] = prediciton[1];
					double res = prediciton[0];
					if (res > maxDerivative)
						maxDerivative = res;
					else if (res < minDerivative)
						minDerivative = res;
					result[i] = res;
					if (res > 1) {
//					System.out.println("probability greater than 1 ahha");
//					System.exit(0);
					}
				} else {

					outputLayerResult[i] = prediciton;
				}

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			i++;
		}
		if (isOutputLayerActivation == false) {
			INDArray output;
			if (layernumber == -1) {

				output = Nd4j.create(result);
				output = output.add((-1 * minDerivative));
				output = output.mul((1 / (maxDerivative - minDerivative)));
				INDArray coeff = Nd4j.create(labelIndexes);
				output = output.mul(interval).add(coeff.mul(interval));
			} else {
				output = Nd4j.create(result);
				INDArray coeff = Nd4j.create(labelIndexes);

//			output = output.mul(interval);

			}
//		result = result.muli(epsilon);

			return new Pair<>(output, null);
		} else
			return new Pair<>(Nd4j.create(outputLayerResult), null);

	}

	@Override
	public INDArray getActivation(INDArray in, boolean training) {

		// in = inputData .* weights 
		Instances trainInstaces = createProperDataset(in.dup(), training);
		double[] result = new double[trainInstaces.size()];

		if (Constants.isEvaluating == false ) {
			activationModel = new J48();

			try {
				activationModel.buildClassifier(trainInstaces);
				isTraind = true;
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
 
		Iterator<Instance> it = trainInstaces.iterator();
		int i = 0;
		double correct = 0d;
		double[][] outputLayerOutput = null;
		if (this.isOutputLayerActivation == true)
			outputLayerOutput = new double[trainInstaces.size()][trainInstaces.numClasses()];
		while (it.hasNext()) {

			double[] prediciton;
			try {
				prediciton = activationModel.predicate(it.next(), isOutputLayerActivation);

				double interval = 1d / trainInstaces.numClasses();
				double res;

				if (isOutputLayerActivation == false) {
//					res = prediciton[0] / 1d * interval + prediciton[1] * interval;
					res = prediciton[0] ;

					if (res == 0)
						res += 0.00001;
					if (res == 1)
						res -= 0.00001;
					result[i] = res;
					if (res > 1) {
						System.out.println("probability greater than 1 ahha");
						System.exit(0);
					}
				} else {

					outputLayerOutput[i] = prediciton;
				}
				if (prediciton[1] == trainInstaces.get(i).classValue())
					correct++;

			} catch (Exception e) {
				e.printStackTrace();
			}

			i++;

		}

		if (isOutputLayerActivation == true)
			return Nd4j.create(outputLayerOutput);
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

			System.out.println(Constants.testInstancesLabel.shapeInfoToString());
			System.out.println(Constants.trainInstancesLabel.shapeInfoToString());
			System.out.println(label.shapeInfoToString());
			System.out.println(in.shapeInfoToString());
			INDArray dataset = Nd4j.concat(1, in, label);
			
			INDArray tempArr;
			Nd4j.shuffle(dataset, 1);
			tempArr =  dataset.get(NDArrayIndex.interval(0, 20), NDArrayIndex.all());
//			System.out.println(dataset.shapeInfoToString());
			instances = _utils.ndArrayToInstances(tempArr);
			
			
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
