
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

package neuralnetwork;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import sigmoid.SigmoidLayer;
import utils.Constants;

import java.util.Collection;
import java.util.Map;

/**
 * Dense layer: a standard fully connected feed forward layer
 */

public class CustomLayer extends FeedForwardLayer {
    private boolean hasBias = true;

    private CustomLayer(Builder builder) {
        super(builder);
        this.hasBias = builder.hasBias;

        initializeConstraints(builder);
    }
 
    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("CustomLayer", getLayerName(), layerIndex, getNIn(), getNOut());
        
        BayesTreeLayer ret = new BayesTreeLayer(conf, Constants.train, Constants.test,Integer.parseInt(layerName.substring(layerName.length()-1)));
 
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return DefaultParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        long numParams = initializer().numParams(this);
        int updaterStateSize = (int) getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if (getIDropout() != null) {
            if (false) {
                //TODO drop connect
                //Dup the weights... note that this does NOT depend on the minibatch size...
                trainSizeVariable += 0; //TODO
            } else {
                //Assume we dup the input
                trainSizeVariable += inputType.arrayElementsPerExample();
            }
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by activation function backprop
        // then we have 'epsilonNext' which is equivalent to input size
        trainSizeVariable += outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, CustomLayer.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize)
                        .workingMemory(0, 0, trainSizeFixed, trainSizeVariable) //No additional memory (beyond activations) for inference
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
                        .build();
    }

    public boolean hasBias(){
        return hasBias;
    }

    
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        private boolean hasBias = true;

        /**
         * If true (default): include bias parameters in the model. False: no bias.
         *
         * @param hasBias If true: include bias parameters in this model
         */
        public Builder hasBias(boolean hasBias){
            this.hasBias = hasBias;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public CustomLayer build() {
            return new CustomLayer(this);
        }
    }

}
