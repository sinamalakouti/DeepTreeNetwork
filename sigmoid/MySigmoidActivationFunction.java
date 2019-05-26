package sigmoid;

import java.util.Enumeration;
import java.util.Iterator;

import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.NDArrayUtil;

import utils.Constants;
import utils._utils;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WekaException;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class MySigmoidActivationFunction extends BaseActivationFunction {

	private J48 activationModel;
	int layernumber = 0;
	boolean isTraind = false;
	Boolean isOutputLayerActivation = false;
//	public BayesTreeActivationFunction(Instances trainInstances, Instances testInstances, boolean isFirstLayer) {
////		testInstancesLabel = NDArrayUtil.toNDArray(_utils.getLabels(testInstances)).transpose();
////		trainInstancesLabel = NDArrayUtil.toNDArray(_utils.getLabels(trainInstances)).transpose();
//	}

	public MySigmoidActivationFunction(int layerNUmber, boolean isOutpuLayerActivation) {
		this.layernumber = layerNUmber;
		this.isOutputLayerActivation = isOutpuLayerActivation;

	}

	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon1) {
//		INDArray n = in.sum(1);
//
//        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new SigmoidDerivative(n));


//        return new Pair<>(dLdz, null);

		  INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new SigmoidDerivative(in));
	        dLdz.muli(epsilon1);
	        return new Pair<>(dLdz, null);
	}

	@Override
	public INDArray getActivation(INDArray in, boolean training) {
//		INDArray n = in.sum(1);
        Nd4j.getExecutioner().execAndReturn(new Sigmoid(in));
        
        System.out.println(in.shapeInfoToString());
        INDArray temp = in.dup();
        
        
		return temp	;
	}

//	private Instances createProperDataset(INDArray in, boolean training) {}
}
