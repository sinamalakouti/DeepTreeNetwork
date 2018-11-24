/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package neuralnetwork.HoeffdingTree;

import java.lang.reflect.Constructor;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import neuralnetwork.CustomLayer;
import utils.Constants;
import weka.core.Instances;

/**
 * A layer with parameters
 * 
 * @author Adam Gibson
 */
public class HoeffdingTreeLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.BaseLayer>
		extends AbstractLayer<CustomLayer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	protected INDArray paramsFlattened;
	protected INDArray gradientsFlattened;
	protected Map<String, INDArray> params;
	protected transient Map<String, INDArray> gradientViews;
	protected double score = 0.0;
	protected ConvexOptimizer optimizer;
	protected Gradient gradient;
	protected Solver solver;

	protected int LayerNumber;
	protected HashMap<Integer, HoeffdingTreeActivationFunction> activationModels = new HashMap<>();

	protected Map<String, INDArray> weightNoiseParams = new HashMap<>();

	public HoeffdingTreeLayer(NeuralNetConfiguration conf, int layernumber) {
		super(conf);
		this.LayerNumber = layernumber;

	}

	// public BayesTreeLayer(NeuralNetConfiguration conf, INDArray input,
	// Instances train, Instances test, int layerNumber) {
	// this(conf, train, test);
	// this.input = input;
	// this.LayerNumber = layerNumber;
	// }

	public CustomLayer layerConf() {
		return (CustomLayer) this.conf.getLayer();
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
		assertInputSet(true);

		INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, true, workspaceMgr);
		INDArray z = bacpropOutput(true, workspaceMgr, W); // Note: using
															// preOutput(INDArray)
															// can't be used as
															// this does a

		double[][] zprim = W.toDoubleMatrix();
		for (int i = 0; i < zprim.length; i++)
			for (int j = 0; j < zprim[i].length; j++) {
				if (Double.isNaN(zprim[i][j]) || Double.isInfinite(zprim[i][j])) {
					System.out.println("stop stage number zero");
					System.out.println(zprim[i][j]);
					System.exit(0);
				}
			}
		zprim = z.toDoubleMatrix();
		for (int i = 0; i < zprim.length; i++)
			for (int j = 0; j < zprim[i].length; j++) {
				if (Double.isNaN(zprim[i][j]) || Double.isInfinite(zprim[i][j])) {
					System.out.println("stop stage number one");
					System.out.println(zprim[i][j]);
					System.exit(0);
				}
			}

		zprim = epsilon.toDoubleMatrix();
		for (int i = 0; i < zprim.length; i++)
			for (int j = 0; j < zprim[i].length; j++) {
				if (Double.isNaN(zprim[i][j]) || Double.isInfinite(zprim[i][j])) {
					System.out.println("stop stage number two");
					System.out.println(zprim[i][j]);
					System.exit(0);
				}
			}
		INDArray delta = z.muli(epsilon).dup();

		zprim = delta.toDoubleMatrix();
		for (int i = 0; i < zprim.length; i++)
			for (int j = 0; j < zprim[i].length; j++) {
				if (Double.isNaN(zprim[i][j]) || Double.isInfinite(zprim[i][j])) {
					System.out.println("stop stage number three");
					System.out.println(zprim[i][j]);
					System.exit(0);
				}
			}

		if (maskArray != null) {
			applyMask(delta);
		}

		Gradient ret = new DefaultGradient();

		INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY); // f
																						// order
		Nd4j.gemm(input, delta, weightGrad, true, false, 1.0, 0.0);

		ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
		INDArray epsilonNext = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD,
				new long[] { W.size(0), delta.size(0) }, 'f');

		zprim = W.toDoubleMatrix();

		for (int i = 0; i < zprim.length; i++)
			for (int j = 0; j < zprim[i].length; j++) {
				if (Double.isNaN(zprim[i][j]) || Double.isInfinite(zprim[i][j])) {
					System.out.println("stop stage number seven");
					System.out.println(zprim[i][j]);

					System.exit(0);
				}
			}
		epsilonNext = W.mmuli(delta.transpose(), epsilonNext).transpose();

		zprim = W.toDoubleMatrix();

		for (int i = 0; i < zprim.length; i++)
			for (int j = 0; j < zprim[i].length; j++) {
				if (Double.isNaN(zprim[i][j]) || Double.isInfinite(zprim[i][j])) {
					System.out.println("stop stage number eight");
					System.out.println(zprim[i][j]);
					System.exit(0);
				}
			}

		weightNoiseParams.clear();

		epsilonNext = backpropDropOutIfPresent(epsilonNext);
		return new Pair<>(ret, epsilonNext);
	}

	public void fit() {
		throw new UnsupportedOperationException("Not supported");
	}

	@Override
	public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
		if (this.input == null)
			return;

		INDArray output = activate(true, workspaceMgr);
		setScoreWithZ(output);
	}

	protected void setScoreWithZ(INDArray z) {
	}

	/**
	 * Objective function: the specified objective
	 * 
	 * @return the score for the objective
	 */

	@Override
	public double score() {
		return score;
	}

	@Override
	public Gradient gradient() {
		return gradient;
	}

	@Override
	public void update(Gradient gradient) {
		for (String paramType : gradient.gradientForVariable().keySet()) {
			update(gradient.getGradientFor(paramType), paramType);
		}
	}

	@Override
	public void update(INDArray gradient, String paramType) {
		setParam(paramType, getParam(paramType).addi(gradient));
	}

	@Override
	public ConvexOptimizer getOptimizer() {
		if (optimizer == null) {
			Solver solver = new Solver.Builder().model(this).configure(conf()).build();
			this.optimizer = solver.getOptimizer();
		}
		return optimizer;
	}

	/**
	 * Returns the parameters of the neural network as a flattened row vector
	 * 
	 * @return the parameters of the neural network
	 */
	@Override
	public INDArray params() {
		return paramsFlattened;
	}

	@Override
	public INDArray getParam(String param) {
		return params.get(param);
	}

	@Override
	public void setParam(String key, INDArray val) {
		if (params.containsKey(key))
			params.get(key).assign(val);
		else
			params.put(key, val);
	}

	@Override
	public void setParams(INDArray params) {
		if (params == paramsFlattened)
			return; // no op
		setParams(params, 'f');
	}

	protected void setParams(INDArray params, char order) {
		List<String> parameterList = conf.variables();
		int length = 0;
		for (String s : parameterList)
			length += getParam(s).length();
		if (params.length() != length)
			throw new IllegalArgumentException("Unable to set parameters: must be of length " + length
					+ ", got params of length " + params.length() + " - " + layerId());
		int idx = 0;
		Set<String> paramKeySet = this.params.keySet();
		for (String s : paramKeySet) {
			INDArray param = getParam(s);
			INDArray get = params.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, idx + param.length()));
			if (param.length() != get.length())
				throw new IllegalStateException("Parameter " + s + " should have been of length " + param.length()
						+ " but was " + get.length() + " - " + layerId());
			param.assign(get.reshape(order, param.shape())); // Use assign due
																// to backprop
																// params being
																// a view of a
			// larger array
			idx += param.length();
		}
	}

	@Override
	public void setParamsViewArray(INDArray params) {
		if (this.params != null && params.length() != numParams())
			throw new IllegalArgumentException("Invalid input: expect params of length " + numParams()
					+ ", got params of length " + params.length() + " - " + layerId());

		this.paramsFlattened = params;
	}

	@Override
	public INDArray getGradientsViewArray() {
		return gradientsFlattened;
	}

	@Override
	public void setBackpropGradientsViewArray(INDArray gradients) {
		if (this.params != null && gradients.length() != numParams())
			throw new IllegalArgumentException("Invalid input: expect gradients array of length " + numParams(true)
					+ ", got array of length " + gradients.length() + " - " + layerId());

		this.gradientsFlattened = gradients;
		this.gradientViews = conf.getLayer().initializer().getGradientsFromFlattened(conf, gradients);
	}

	@Override
	public void setParamTable(Map<String, INDArray> paramTable) {
		this.params = paramTable;
	}

	@Override
	public Map<String, INDArray> paramTable() {
		return paramTable(false);
	}

	@Override
	public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
		return params;
	}

	/**
	 * Get the parameter, after applying any weight noise (such as DropConnect)
	 * if necessary. Note that during training, this will store the post-noise
	 * parameters, as these should be used for both forward pass and backprop,
	 * for a single iteration. Consequently, the parameters (post noise) should
	 * be cleared after each training iteration
	 *
	 * @param param
	 *            Parameter key
	 * @param training
	 *            If true: during training
	 * @return The parameter, after applying any noise
	 */
	protected INDArray getParamWithNoise(String param, boolean training, LayerWorkspaceMgr workspaceMgr) {
		INDArray p;
		if (layerConf().getWeightNoise() != null) {
			if (training && weightNoiseParams.size() > 0 && weightNoiseParams.containsKey(param)) {
				// Re-use these weights for both forward pass and backprop -
				// don't want to use 2
				// different params here
				// These should be cleared during backprop
				return weightNoiseParams.get(param);
			} else {
				try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
					p = layerConf().getWeightNoise().getParameter(this, param, getIterationCount(), getEpochCount(),
							training, workspaceMgr);
				}
			}

			if (training) {
				// Store for re-use in backprop
				weightNoiseParams.put(param, p);
			}
		} else {
			return getParam(param);
		}

		return p;
	}

	protected INDArray preOutput(boolean training, LayerWorkspaceMgr workspaceMgr) {
		assertInputSet(false);
		applyDropOutIfNecessary(training, workspaceMgr);
		INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, training, workspaceMgr);

		// normalization:

//		 double mu = W.meanNumber().doubleValue();
//		 double std = Math.sqrt(W.varNumber().doubleValue());
//		 W = W.subi(mu);
//		 W = W.divi(std);

		double[][] zprim = W.toDoubleMatrix();
		for (int i = 0; i < zprim.length; i++)
			for (int j = 0; j < zprim[i].length; j++) {
				if (Double.isNaN(zprim[i][j]) || Double.isInfinite(zprim[i][j])) {
					System.out.println("stop stage number four");
					System.out.println(zprim[i][j]);
					System.exit(0);
				}
			}

		// Input validation:
		if (input.rank() != 2 || input.columns() != W.rows()) {
			if (input.rank() != 2) {
				throw new DL4JInvalidInputException("Input that is not a matrix; expected matrix (rank 2), got rank "
						+ input.rank() + " array with shape " + Arrays.toString(input.shape())
						+ ". Missing preprocessor or wrong input type? " + layerId());
			}
			throw new DL4JInvalidInputException(
					"Input size (" + input.columns() + " columns; shape = " + Arrays.toString(input.shape())
							+ ") is invalid: does not match layer input size (layer # inputs = " + W.size(0) + ") "
							+ layerId());
		}

		INDArray z = null;
		INDArray result = null;

		for (int neuron = 0; neuron < W.columns(); neuron++) {

			if (!activationModels.containsKey(neuron))
				activationModels.put(neuron, new HoeffdingTreeActivationFunction(this.LayerNumber, false, neuron));

			INDArray weight = W.getColumn(neuron);

			z = input.mulRowVector(weight.transpose());

			zprim = z.toDoubleMatrix();
			for (int i = 0; i < zprim.length; i++)
				for (int j = 0; j < zprim[i].length; j++) {
					if (Double.isNaN(zprim[i][j]) || Double.isInfinite(zprim[i][j])) {
						System.out.println("stop stage number five");
						System.out.println("at neuron : \t" + neuron);
						System.out.println("layer number is \t" + LayerNumber);
						System.out.println("number of features\t " + input.shapeInfoToString());
						System.out.println(weight);
						System.out.println(zprim[i][j]);
						System.exit(0);
					}
				}

			if (neuron == 0) {

				INDArray ztemp;
				if (LayerNumber == 0)
					ztemp = z.getColumns(Constants.attributesIndexes.get(neuron)).dup();
				else
					ztemp = z.dup();

				INDArray ret = activationModels.get(neuron).getActivation(ztemp, training);

				result = ret.transpose().dup();
			} else {
				INDArray ztemp;
				if (LayerNumber == 0)
					ztemp = z.getColumns(Constants.attributesIndexes.get(neuron)).dup();
				else
					ztemp = z.dup();

				INDArray ret = activationModels.get(neuron).getActivation(ztemp, training);

				result = Nd4j.concat(1, result, ret.transpose());
			}

		}
		double avgDepth =0d;
		for ( int neuron = 0 ; neuron < Constants.numberOfNeurons ; neuron++ )
			avgDepth += (double) activationModels.get(neuron).getActivationModel().treeDepth ;
		
		avgDepth = avgDepth / Constants.numberOfNeurons;
		Constants.avgHFDepth[this.LayerNumber] = avgDepth;
		if (maskArray != null) {
			applyMask(z);
		}

		return result;
	}

	@Override
	public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
		INDArray z = preOutput(training, workspaceMgr);
		INDArray ret = z.dup();

		return ret;
	}

	protected INDArray bacpropOutput(boolean training, LayerWorkspaceMgr workspaceMgr, INDArray W) {
		assertInputSet(false);
		applyDropOutIfNecessary(true, workspaceMgr);
		// Input validation:
		if (input.rank() != 2 || input.columns() != W.rows()) {
			if (input.rank() != 2) {
				throw new DL4JInvalidInputException("Input that is not a matrix; expected matrix (rank 2), got rank "
						+ input.rank() + " array with shape " + Arrays.toString(input.shape())
						+ ". Missing preprocessor or wrong input type? " + layerId());
			}
			throw new DL4JInvalidInputException(
					"Input size (" + input.columns() + " columns; shape = " + Arrays.toString(input.shape())
							+ ") is invalid: does not match layer input size (layer # inputs = " + W.size(0) + ") "
							+ layerId());
		}

		INDArray z = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.size(0), W.size(0));
		INDArray result = null;
		for (int neuron = 0; neuron < W.columns(); neuron++) {

			INDArray weight = W.getColumn(neuron);
			z.assign(input.mulRowVector(weight.transpose()));

			if (neuron == 0) {

				INDArray ztemp;
				if (LayerNumber == 0)
					ztemp = z.getColumns(Constants.attributesIndexes.get(neuron)).dup();
				else
					ztemp = z;

				Pair<INDArray, INDArray> ret = activationModels.get(neuron).backprop(ztemp, null);
				result = ret.getFirst().dup();
			} else {
				INDArray ztemp;
				if (LayerNumber == 0)
					ztemp = z.getColumns(Constants.attributesIndexes.get(neuron)).dup();
				else
					ztemp = z.dup();

				Pair<INDArray, INDArray> ret = activationModels.get(neuron).backprop(ztemp, null);
				INDArray tmp = ret.getFirst().dup();
				result = Nd4j.concat(1, result, tmp);
			}

		}

		if (maskArray != null) {
			applyMask(z);
		}

		return result;
	}

	@Override
	public double calcL2(boolean backpropParamsOnly) {
		double l2Sum = 0.0;
		for (Map.Entry<String, INDArray> entry : paramTable().entrySet()) {
			double l2 = layerConf().getL2ByParam(entry.getKey());
			if (l2 > 0) {

				double norm2 = getParam(entry.getKey()).norm2Number().doubleValue();
				l2Sum += 0.5 * l2 * norm2 * norm2;
			}
		}
		return l2Sum;
	}

	@Override
	public double calcL1(boolean backpropParamsOnly) {
		double l1Sum = 0.0;
		for (Map.Entry<String, INDArray> entry : paramTable().entrySet()) {
			double l1 = layerConf().getL1ByParam(entry.getKey());
			if (l1 > 0) {
				double norm1 = getParam(entry.getKey()).norm1Number().doubleValue();
				l1Sum += l1 * norm1;
			}
		}

		return l1Sum;
	}

	@Override
	public Layer clone() {
		Layer layer = null;
		try {
			Constructor c = getClass().getConstructor(NeuralNetConfiguration.class);
			layer = (Layer) c.newInstance(conf);
			Map<String, INDArray> linkedTable = new LinkedHashMap<>();
			for (Map.Entry<String, INDArray> entry : params.entrySet()) {
				linkedTable.put(entry.getKey(), entry.getValue().dup());
			}
			layer.setParamTable(linkedTable);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return layer;

	}

	/**
	 * The number of parameters for the model
	 *
	 * @return the number of parameters for the model
	 */
	@Override
	public int numParams() {
		int ret = 0;
		for (INDArray val : params.values())
			ret += val.length();
		return ret;
	}

	@Override
	public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
		if (input != null) {
			setInput(input, workspaceMgr);
			applyDropOutIfNecessary(true, workspaceMgr);
		}
		if (solver == null) {
			solver = new Solver.Builder().model(this).configure(conf()).listeners(getListeners()).build();
		}
		this.optimizer = solver.getOptimizer();
		solver.optimize(workspaceMgr);
	}

	@Override
	public String toString() {

		return getClass().getName() + "{" + "conf=" + conf + ", score=" + score + ", optimizer=" + optimizer
				+ ", listeners=" + trainingListeners + '}';
	}

	@Override
	public void clear() {
		super.clear();
		weightNoiseParams.clear();
	}

	@Override
	public void clearNoiseWeightParams() {
		weightNoiseParams.clear();
	}

	/**
	 * Does this layer have no bias term? Many layers (dense, convolutional,
	 * output, embedding) have biases by default, but no-bias versions are
	 * possible via configuration
	 *
	 * @return True if a bias term is present, false otherwise
	 */

	// check and implement this function and the consequences
	public boolean hasBias() {
		// Overridden by layers supporting no bias mode: dense, output,
		// convolutional,
		// embedding
		// return true;
		// todo : check that it should be true or false
		return false;
	}

	@Override
	public boolean isPretrainLayer() {
		return false;
	}
}
