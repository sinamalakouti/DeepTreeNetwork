package neuralnetwork;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import onnx.OnnxProto3.AttributeProto;
import onnx.OnnxProto3.GraphProto;
import onnx.OnnxProto3.NodeProto;
import weka.dl4j.activations.ActivationSoftmax;

@JsonInclude(JsonInclude.Include.NON_NULL)

public class LossBayesTree extends DifferentialFunction implements Serializable, ILossFunction {

	protected final INDArray weights;

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public LossBayesTree(INDArray weights) {
		if (weights != null && !weights.isRowVector()) {
			throw new IllegalArgumentException("Weights array must be a row vector");
		}
		this.weights = weights;
	}

	@Override
	public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask,
			boolean average) {
//		System.out.println("compute score");
		INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
		double score = -scoreArr.sumNumber().doubleValue();
		double s[][] = scoreArr.toDoubleMatrix();
		double sd = 0d;
		for (int i = 0; i < s.length; i++)
			for (int j = 0; j < s[0].length; j++) {
				if (Double.isNaN(s[i][j])) {
					System.out.println(s[i][j]);
					System.out.println("ey babaa chi kar konim ");
				}
				sd += s[i][j];
			}

		if (average) {
			score /= scoreArr.size(0);
		}

		return score;
	}

	@Override
	public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		System.out.println("compute Score array");

		System.exit(0);
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
//		System.out.println("compute gradients");

		INDArray grad;

		INDArray output = activationFn.getActivation(preOutput.dup(), true);

		if (mask != null && LossUtil.isPerOutputMasking(output, mask)) {
			throw new UnsupportedOperationException("Per output masking for MCXENT + softmax: not supported");
		}

		// Weighted loss function
		if (weights != null) {
			if (weights.length() != output.size(1)) {
				throw new IllegalStateException("Weights vector (length " + weights.length()
						+ ") does not match output.size(1)=" + output.size(1));
			}
			
			INDArray temp = labels.mulRowVector(weights);
			INDArray col = temp.sum(1);
			grad = output.mulColumnVector(col).sub(temp);
			System.exit(0);
		} else {
			grad = output.subi(labels);
		}
		
		
//        grad = activationFn.backprop(preOutput, dLda).getFirst(); //TODO activation function with weights

		
		
		 //Loss function with masking
        if (mask != null) {
            LossUtil.applyMask(grad, mask);
        }

        
        return grad;
        }

	@Override
	public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn,
			INDArray mask, boolean average) {
		System.out.println("compute gradient and score");
		System.exit(0);

		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String name() {
		System.out.println("getName");
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public SDVariable[] outputVariables(String baseName) {
		System.out.println("SD Variable");
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<SDVariable> doDiff(List<SDVariable> f1) {
		System.out.println("Do Diff");
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode,
			GraphDef graph) {

		System.out.println("initFromTensorflow");
		// TODO Auto-generated method stub

	}

	@Override
	public void initFromOnnx(NodeProto node, SameDiff initWith, Map<String, AttributeProto> attributesForNode,
			GraphProto graph) {
		
		// TODO Auto-generated method stub

	}

	@Override
	public String onnxName() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String tensorflowName() {
		// TODO Auto-generated method stub
		return null;
	}

	private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
		if (labels.size(1) != preOutput.size(1)) {
			throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1)
					+ ") does not match output layer" + " number of outputs (nOut = " + preOutput.size(1) + ") ");

		}

		INDArray output = activationFn.getActivation(preOutput.dup(), true);
		if (activationFn instanceof ActivationSoftmax) {
			System.out.println("shit shooot soft max");
			System.exit(0);
		}
		double s[][] = output.toDoubleMatrix();
		
		for (int i = 0; i < s.length; i++)
			for (int j = 0; j < s[0].length; j++) {
				if (Double.isNaN(s[i][j])) {
					System.out.println(s[i][j]);
					System.out.println("ey babaa chi kar konim ");
				}

				if (s[i][j] == 0) {
					s[i][j] += 0.0001;

				}
				if ( s[i][j] == 1)
				{
					s[i][j] -= 0.0001;
				}
			}
		output = Nd4j.create(s);
		INDArray scoreArr1= Transforms.log(output, false).mul(labels);
		INDArray scoreArr2 = Transforms.log((output.sub(1)).mul(-1),false).mul((labels.sub(1)).mul(-1));
		INDArray scoreArr = (scoreArr1.add(scoreArr2)).mul(-1);	
		
		s = scoreArr.toDoubleMatrix();
		for (int i = 0; i < s.length; i++)
			for (int j = 0; j < s[0].length; j++) {
				if (Double.isNaN(s[i][j])) {
					System.out.println(s[i][j]);
					System.out.println("ey babaa chi kar konim ");
					System.exit(0);
				}
			}

		// Weighted loss function
		if (weights != null) {
			if (weights.length() != scoreArr.size(1)) {
				throw new IllegalStateException("Weights vector (length " + weights.length()
						+ ") does not match output.size(1)=" + preOutput.size(1));
			}
			scoreArr.muliRowVector(weights);
		}

		if (mask != null) {
			LossUtil.applyMask(scoreArr, mask);
		}
		return scoreArr;
	}
}
