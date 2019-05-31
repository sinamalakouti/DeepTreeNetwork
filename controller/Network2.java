package controller;

import neuralnetwork.CustomLayer;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import utils.Constants;
import utils._utils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.pmml.Constant;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.*;
import java.util.*;

public class Network2 {

    private static Logger log = LoggerFactory.getLogger(Network2.class);


    DataSetIterator mnistTrain;
    DataSetIterator mnistTest;
    DataSet tempTrainSet;
    Instances trainSet2 = null;

    int iteration_based;

    public static void main(String[] args) throws Exception {

        // TODO Nd4j.setDataType(Type.DOUBLE);


        boolean deSerializing= true;
        boolean serializing = true;

        Constants.weightLayerMin = new double[2];
        Constants.weightLayerMin[0] = Double.POSITIVE_INFINITY;
        Constants.weightLayerMin[1] = Double.POSITIVE_INFINITY;
        Constants.weightLayerMax = new double[2];
        Constants.weightLayerMax[0] = Double.NEGATIVE_INFINITY;
        Constants.weightLayerMax[1] = Double.NEGATIVE_INFINITY;

        final int numInputs = 784;
        int outputNum = 10;
        log.info("Build model....");
        Constants.numberOfLayers = 2;
        Constants.numberOfNeurons = 10;
        Constants.batchSize = 100;
        Constants.avgHFDepth = new double[Constants.numberOfLayers];
        double numberTrainExamples = 60000d;
        Constants.numBatches = (int) ((numberTrainExamples) / Constants.batchSize);
        Constants.numClasses = 10;
        Constants.maximumDepth = 20;
        Constants.maximumDepth--;

        int feature_ratio = 60;


        Network2 net2 = new Network2();
        if (deSerializing == false) {
            net2.init_problem_configuration(numInputs, feature_ratio);
            net2.save_problem_configuration(numInputs, feature_ratio);
        }
        else
            net2.load_problem_configuration(numInputs, feature_ratio);


        DataNormalization scaler = new NormalizerStandardize();
        scaler.fit(net2.mnistTrain);
        net2.mnistTrain.setPreProcessor(scaler);
        net2.mnistTest.setPreProcessor(scaler); // same normalization for better results
        net2.mnistTrain.reset();


        int counter = 0;
        Instances trainTemp = null;
        int c = 0;

        while (net2.mnistTrain.hasNext()) {
            DataSet set = net2.mnistTrain.next();
            if (c == 0) {
                net2.trainSet2 = _utils.dataset2Instances(set);
            } else {
                trainTemp = _utils.dataset2Instances(set);
                for (int i = 0; i < trainTemp.size(); i++)
                    net2.trainSet2.add(trainTemp.get(i));
            }

            c++;
        }
        net2.mnistTrain.reset();

        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "" + (net2.trainSet2.numAttributes()); // range of variables to
        convert.setOptions(options);
        convert.setInputFormat(net2.trainSet2);
        net2.trainSet2 = weka.filters.Filter.useFilter(net2.trainSet2, convert);
        net2.trainSet2.setClassIndex(net2.trainSet2.numAttributes() - 1);
        net2.tempTrainSet = _utils.instancesToDataSet(net2.trainSet2);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(6)

                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .weightInit(WeightInit.XAVIER).updater(new Sgd(0.1)).l2(1e-4).list()
                // new BayesTreeActivationFunction(0, false, -1198)

                .layer(0,
                        new CustomLayer.Builder().nIn(numInputs).nOut(Constants.numberOfNeurons)
                                .activation(Activation.SIGMOID).build())
                .layer(1,
                        new CustomLayer.Builder().nIn(Constants.numberOfNeurons).nOut(Constants.numberOfNeurons)
                                .activation(Activation.SIGMOID).build())
                .layer(2,
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX).nIn(Constants.numberOfNeurons).nOut(outputNum).build())
                .backprop(true).pretrain(false).build();

        // run the model
        Constants.model = new MultiLayerNetwork(conf);

        Constants.model.init();
        Constants.model.setListeners(new ScoreIterationListener(5));
        System.out.println("start");


//			tmp1.clear();


        System.out.println("NETWORK2.JAVA is running   784 / 2 and 20 neurons");
        for (int i = 0 + net2.iteration_based; i < 156; i++) {
            // in the first iteration do the bagging test and the each batch
            // test :D
            for (int b = 0; b < Constants.numBatches; b++) {

                DataSet set = getBatchTrainSet(b, Constants.batchSize, net2.tempTrainSet, net2.trainSet2);

                if (i % 50 == 0 && serializing)  {
                    Constants.isSerialzing = true;
                    _utils.serializing();
                    File file = new File("problem/problem_configuration");
                    FileWriter fr = new FileWriter(file);
                    BufferedWriter out = new BufferedWriter(fr);
                    String str = new String();
                    str += "iteration_based " + net2.iteration_based + "\n";
                    out.write(str);
                    out.close();
                    fr.close();
                }
                if ( deSerializing == true){
                    Constants.isDeSerializing = true;
                    _utils.deserializing();
                }


                Constants.model.fit(set);

                if (Constants.isSerialzing == true)
                    Constants.isSerialzing = false;
                if (Constants.isDeSerializing == true) {
                    Constants.isDeSerializing = false;
                    deSerializing = false;
                }


            }

            if (i % 2 == 0) {
                Constants.isEvaluating = true;
                log.info("Evaluate model....");
                //
                Evaluation eval = new Evaluation(outputNum); // create an
                //
                while (net2.mnistTest.hasNext()) {

                    DataSet next = net2.mnistTest.next();
                    System.out.println(Constants.isEvaluating);
                    _utils.setLabels(next.getLabels(), Constants.isEvaluating,
                            false);
                    INDArray output = Constants.model.output(next.getFeatures());

                    eval.eval(next.getLabels(), output);
                }
                net2.mnistTest.reset();
                //
                String path =
                        "/root/research/result/phase4/randomClassConfig/23/resultIteration_" + i;
                File file = new File(path);
                BufferedWriter out = new BufferedWriter(new
                        FileWriter(file));
                String avglayersTreesDepth = "";
                for (int l = 0; l < Constants.numberOfLayers; l++)
                    avglayersTreesDepth = avglayersTreesDepth + " " +
                            Constants.avgHFDepth[l];
                out.write(eval.stats() + "\nerrors\t" + Constants.model.score() + "\n" + avglayersTreesDepth);
//
                System.out.println(eval.stats() + "\n" + "errors:  " +
                        Constants.model.score() + "\n" + avglayersTreesDepth);

                //
                out.close();
                Constants.isEvaluating = false;
                //
            }
            // if ( i == 10 ){
            // _utils.draw_accuracy_fscore("hello world plot", "", 0, 10);
            // }

        }

        Constants.isEvaluating = true;
        Evaluation eval = new Evaluation(outputNum); // create an evaluation
        // object with 10
        // possible classes
        counter = 0;
        while (net2.mnistTest.hasNext()) {

            DataSet next = net2.mnistTest.next();
            _utils.setLabels(next.getLabels(), Constants.isEvaluating, false);
            INDArray output = Constants.model.output(next.getFeatures());
            eval.eval(next.getLabels(), output); // check the prediction
            // against the true
            // class
            counter++;
        }
//			System.out.println(counter);
        log.info(eval.stats());

    }


    private static DataSet getBatchTrainSet(int batchNumber, int batchRate, DataSet trainSet, Instances training) {

        INDArray features = trainSet.getFeatures();
        INDArray labels = trainSet.getLabels();
        int start = batchNumber * batchRate;
        int end = (batchNumber + 1) * batchRate;

        INDArray batchTrain_features = features.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
        INDArray batchTrain_labels = labels.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());

        DataSet set = new DataSet(batchTrain_features, batchTrain_labels);
        List<Instance> list = training.subList(start, end);
        double[] labels_list = new double[list.size()];
        for (int i = 0; i < list.size(); i++)
            labels_list[i] = list.get(i).classValue();
        Constants.trainInstancesLabel = Nd4j.create(labels_list).transpose();

        features.cleanup();
        labels.cleanup();
        batchTrain_features.cleanup();
        batchTrain_labels.cleanup();


        return set;

    }

    private void init_problem_configuration(int numInputs, int feature_ratio) throws Exception {
        System.out.println("INITING THE PROBLEM");
        ArrayList<Integer> featuresVector = new ArrayList<>();
        for (int i = 0; i < 784; i++)
            featuresVector.add(i);
        /**
         *
         * TODO : here we set the number of the attributes that is going to be
         * chose for each neurons ( max number) => we need to find some
         * automatic way
         *
         *
         */

        int max = numInputs / feature_ratio;
        for (int j = 0; j < Constants.numberOfNeurons; j++) {
            Collections.shuffle(featuresVector);
            int[] temp = new int[max];
            for (int i = 0; i < max; i++) {
                temp[i] = featuresVector.get(i);
            }

            Constants.attributesIndexes.put(j, temp);

        }

        featuresVector.clear();

        // class configuration for each neuron

        ArrayList<Integer> tmp1 = new ArrayList<Integer>();
//
        for (int c = 0; c < Constants.numClasses - 1; c++) {
            // for 4 classes -> it is set only for mnist dataset ( to be changed
            // )
            for (int i = 0; i < (int) (Constants.numberOfNeurons / Constants.numClasses); i++) {
                tmp1.add(c);
            }
        }
//
        while (tmp1.size() < Constants.numberOfNeurons) {
//				Random rand = new Random();
//				int i = rand.nextInt(10);
//
            tmp1.add(Constants.numClasses - 1);
        }
//	TODO : CHANGE IT BACK. ASAP
        for (int l = 0; l < Constants.numberOfLayers; l++) {


            @SuppressWarnings("unchecked")
            ArrayList<Integer> tmp2 = (ArrayList<Integer>) tmp1.clone();
            Collections.shuffle(tmp2);
            Constants.classChosedArray.put(l, tmp2);
        }

        // set-up the project :

        mnistTrain = new MnistDataSetIterator(Constants.batchSize, true, 6);
        mnistTest = new MnistDataSetIterator(10000, false, 6);


        iteration_based = 0;

    }


    private void save_problem_configuration(int numInputs, int feature_ratio) throws IOException {

        System.out.println("SAVING THE PROBLEM");
        File file = new File("problem/problem_configuration");
        FileWriter fr = new FileWriter(file);
        BufferedWriter out = new BufferedWriter(fr);
        String str = new String();
        str += "iteration_based " + iteration_based + "\n";
        out.write(str);

        FileOutputStream attributesIndexes_file =
                new FileOutputStream("problem/attributesIndexes.ser");
        ObjectOutputStream attIndex_out = new ObjectOutputStream(attributesIndexes_file);
        attIndex_out.writeObject(Constants.attributesIndexes);


        FileOutputStream class_file =
                new FileOutputStream("problem/class_file.ser");
        ObjectOutputStream class_file_out = new ObjectOutputStream(class_file);
        class_file_out.writeObject(Constants.classChosedArray);

        FileOutputStream mnistTrain_file =
                new FileOutputStream("problem/mnistTrain_file.ser");
        ObjectOutputStream mnistTrain_file_out = new ObjectOutputStream(mnistTrain_file);
        mnistTrain_file_out.writeObject(mnistTrain);


        FileOutputStream mnistTest_file =
                new FileOutputStream("problem/mnistTest_file.ser");
        ObjectOutputStream mnistTest_file_out = new ObjectOutputStream(mnistTest_file);
        mnistTest_file_out.writeObject(mnistTest);


        out.close();
        attributesIndexes_file.close();
        attIndex_out.close();
        class_file.close();
        class_file_out.close();
        mnistTrain_file.close();
        mnistTrain_file_out.close();
        mnistTrain_file.close();
        mnistTrain_file_out.close();
        mnistTest_file.close();
        mnistTest_file_out.close();

    }


    private void load_problem_configuration(int numInputs, int feature_ratio) throws IOException, ClassNotFoundException {

        System.out.println("LOADING THE PROBLEM");
        File file = new File("problem/problem_configuration");

        Scanner in = new Scanner(file);

        String str = in.nextLine();
        String[] arr = str.split(" ");
        this.iteration_based = Integer.parseInt(arr[1]);

        FileInputStream attributesIndexes_file =
                new FileInputStream("problem/attributesIndexes.ser");
        ObjectInputStream attIndex_in = new ObjectInputStream(attributesIndexes_file);

        Constants.attributesIndexes = (HashMap<Integer, int[]>) attIndex_in.readObject();


        FileInputStream class_file =
                new FileInputStream("problem/class_file.ser");
        ObjectInputStream class_file_in = new ObjectInputStream(class_file);
        Constants.classChosedArray = (HashMap<Integer, ArrayList<Integer>>) class_file_in.readObject();

        FileInputStream mnistTrain_file =
                new FileInputStream("problem/mnistTrain_file.ser");
        ObjectInputStream mnistTrain_file_in = new ObjectInputStream(mnistTrain_file);
        mnistTrain = (DataSetIterator) mnistTrain_file_in.readObject();


        FileInputStream mnistTest_file =
                new FileInputStream("problem/mnistTest_file.ser");
        ObjectInputStream mnistTest_file_in = new ObjectInputStream(mnistTest_file);
        mnistTest = (DataSetIterator) mnistTest_file_in.readObject();


        mnistTest_file.close();
        mnistTest_file_in.close();
        mnistTrain_file.close();
        mnistTrain_file_in.close();
        class_file.close();
        class_file_in.close();
        attributesIndexes_file.close();
        attIndex_in.close();


    }
}