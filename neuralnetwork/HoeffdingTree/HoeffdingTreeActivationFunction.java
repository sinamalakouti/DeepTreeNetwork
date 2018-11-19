package neuralnetwork.HoeffdingTree;

import java.util.Enumeration;
import java.util.Iterator;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import utils.Constants;
import utils._utils;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class HoeffdingTreeActivationFunction extends BaseActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6622285329198225782L;
	private HoeffdingTree activationModel;
	int layernumber = 0;
	boolean isTraind = false;
	Boolean isOutputLayerActivation = false;
	int neuronNumber = 0;
	boolean createdHF = false;

	public HoeffdingTreeActivationFunction(int layerNUmber, boolean isOutpuLayerActivation, int neuronNumber) {
		this.layernumber = layerNUmber;
		this.isOutputLayerActivation = isOutpuLayerActivation;
		this.neuronNumber = neuronNumber;

	}

	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon1) {

		// if ( isOutputLayerActivation == true)
		// assertShape(in, epsilon);

		Instances trainInstaces = createProperDataset(in, true);
		double[] result = new double[trainInstaces.size()];

		Enumeration<Instance> it = trainInstaces.enumerateInstances();

		int i = 0;
		double[][] outputLayerResult = null;
		if (isOutputLayerActivation == true)
			outputLayerResult = new double[trainInstaces.size()][trainInstaces.numClasses()];

		while (it.hasMoreElements()) {

			try {
				Instance next = it.nextElement();
				// TODO : add mapping again
				// [Constants.classChosedArray.get(layernumber).get(neuronNumber)]

				double[] predictionDerivative = activationModel.predicate_derivative(next);

				// for mapping : result [i] = predictionDerivative[0]; should
				// changed true -> false in the line above

				if (predictionDerivative.length != Constants.numClasses && Constants.classChosedArray.get(layernumber)
						.get(neuronNumber) >= predictionDerivative.length) {
					System.err.println("inja ham bayad doros shavad dar moshtagh e gerami");
					result[i] = 0;
				} else
					result[i] = predictionDerivative[Constants.classChosedArray.get(layernumber).get(neuronNumber)];

				// if (isOutputLayerActivation == false) {
				// labelIndexes[i] = prediciton[1];
				//// double res = prediciton[0];
				// if (res > maxDerivative)
				// maxDerivative = res;
				// else if (res < minDerivative)
				// minDerivative = res;
				// result[i] = res;
				// if (res > 1) {
				//// System.out.println("probability greater than 1 ahha");
				//// System.exit(0);
				// }
				// } else {

				// outputLayerResult[i] = prediciton;
				// }

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			i++;
		}
		if (isOutputLayerActivation == false) {
			INDArray output;
			if (layernumber == -1) {

				// output = Nd4j.create(result);
				// output = output.add((-1 * minDerivative));
				// output = output.mul((1 / (maxDerivative - minDerivative)));
				// INDArray coeff = Nd4j.create(labelIndexes);
				// output = output.mul(interval).add(coeff.mul(interval));
			} else {
				// output = output.add((-1 * minDerivative));
				// output = output.mul((1 / (maxDerivative - minDerivative)));
				// INDArray coeff = Nd4j.create(labelIndexes);

				// output = output.mul(interval);

			}
			// result = result.muli(epsilon);

			output = Nd4j.create(result).transpose();
			// normalization : ( 0, 1 )

			// double min = output.minNumber().doubleValue();
			// double max = output.maxNumber().doubleValue();
			//
			// output = output.sub(min);
			// output = output.div(max - min);

			return new Pair<>(output, null);
		} else {
			// System.out.println("mage darim ");
			return new Pair<>(Nd4j.create(outputLayerResult), null);
		}
	}

	public INDArray getActivation(INDArray in, boolean training) {

		// in = inputData .* weights

		Instances trainInstaces = createProperDataset(in.dup(), training);
		// if ( hf == null)
		// {
		// hf = new HoeffdingTree();
		// this.createdHF = true;
		// try {
		// hf.buildClassifier(trainInstaces);
		// } catch (Exception e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		//
		// }else{
		// for( int i =0 ; i< trainInstaces.size(); i ++)
		// try {
		// if ( hf == null)
		// System.out.println("shoot");
		// hf.updateClassifier(trainInstaces.get(i));
		//
		// } catch (Exception e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		//
		// }
		//
		// if ( hf.m_root.isLeaf() == false){
		// System.out.println("lets do it my frendinto");
		// try {
		// hf.classifyInstance(trainInstaces.get(0));
		// } catch (Exception e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		// }
		double[] result = new double[trainInstaces.size()];

		// Instances trainInstaces2 = createProperDataset(input.dup(),
		// training);
		// J48 originalTree = new J48();

		if (Constants.isEvaluating == false) {
//			System.out.println("trai /ning");
			if (activationModel == null) {
				activationModel = new HoeffdingTree();
				try {
					this.isTraind = true;

					activationModel.buildClassifier(trainInstaces);
					// originalTree.buildClassifier(trainInstaces2);

					isTraind = true;
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			} else {
				try {
					Iterator<Instance> it = trainInstaces.iterator();
					while(it.hasNext())
						activationModel.updateClassifier(it.next());
				} catch (Exception e) {
					e.printStackTrace();
				}
			}

		}

		Iterator<Instance> it = trainInstaces.iterator();

		int i = 0;
		double[][] outputLayerOutput = null;
		if (this.isOutputLayerActivation == true)
			outputLayerOutput = new double[trainInstaces.size()][trainInstaces.numClasses()];
		while (it.hasNext()) {

			double[] prediction;
			// double[] prediction2;

			try {
				Instance next = it.next();
				
				prediction = activationModel.predicate(next);
				// prediciton2 = activationModel.predicate(next ,
				// isOutputLayerActivation);

				double res;

				if (isOutputLayerActivation == false) {

					if (prediction.length != Constants.numClasses
							&& Constants.classChosedArray.get(layernumber).get(neuronNumber) >= prediction.length) {
						System.err.println("in doostemoon bayad bezoodi hal beshe :))");
						res = 0;
					} else
						res = prediction[Constants.classChosedArray.get(layernumber).get(neuronNumber)];
					result[i] = res;

					if (res < 0 || res - 1 > 0.01) {
						System.out.println("ya khode khoda probablity not valid");
						System.out.println(res);
						System.exit(0);
					}
				} else {
					System.out.println("WHY HERE IN  Loss FUNCTION -> output layer");
					System.exit(0);
					outputLayerOutput[i] = prediction;
				}

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

			if (training == false){
				label = Constants.testInstancesLabel;
			}
			else {
				_utils.setLabels(Constants.model.getLabels(), false, training);
				label = Constants.trainInstancesLabel;

			}
			if (in.rows() != label.rows()) {
				System.out.println(in.shapeInfoToString());
				System.out.println("********");
				System.out.println(label.shapeInfoToString());
			}
			INDArray dataset = Nd4j.concat(1, in, label);

			instances = _utils.ndArrayToInstances(dataset);
			instances.setClassIndex(instances.numAttributes() - 1);

			if (!instances.classAttribute().isNominal()) {
				NumericToNominal convert = new NumericToNominal();
				String[] options = new String[2];
				options[0] = "-R";
				options[1] = "" + (instances.classIndex() + 1); // range of
				// variables to
				// make numeric
				convert.setOptions(options);
				convert.setInputFormat(instances);
				instances = weka.filters.Filter.useFilter(instances, convert);
			}
		} catch (WekaException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return instances;

	}

	// @Override
	// public INDArray getActivation(INDArray in, boolean training) {
	// // TODO Auto-generated method stub
	// return null;
	// }
}
