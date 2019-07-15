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
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
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
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.*;
import java.util.*;

public class Network2 {

    private static Logger log = LoggerFactory.getLogger(Network2.class);


//    DataSetIterator mnistTrain;
//


    KFoldIterator fivefoldIterator = null;
    int iteration_based;
    int fold_iteration;

    public static void main(String[] args) throws Exception {
//        DataSetIterator mnistTrain = new MnistDataSetIterator(60000, true, 6);
//        DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 6);

        kfold_experiment(5);

    }


    private static void kfold_experiment(int k) throws Exception {


        boolean deSerializing = false;
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
        double numberTrainExamples = 70000d / k * 4;
        Constants.allData_size = 70000;
        Constants.numBatches = (int) ((numberTrainExamples) / Constants.batchSize);
        Constants.numClasses = 10;
        Constants.maximumDepth = 20;
        Constants.maximumDepth--;
        Constants.output_file_prefix = "/home/research/result/phase5/9";

        double learning_rate = 0.1;
        int feature_ratio = 20;
        DataSet mnistTrain;
        DataSet mnistTest;
        //
        Instances trainSet2 = null;


        Network2 net2 = new Network2();


        if (deSerializing == false) {
            net2.init_problem_configuration(numInputs, feature_ratio);
            net2.fold_iteration = 0;
            net2.save_kfold(numInputs, feature_ratio, trainSet2, k);

//            DataSet tempTrainSet;
//            tempTrainSet = _utils.instancesToDataSet(trainSet2);
//            saveBatches(Constants.numBatches, Constants.batchSize, tempTrainSet, trainSet2);
        } else
            net2.load_kfold_configuration();


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(6)

                .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .weightInit(WeightInit.XAVIER).updater(new Sgd(learning_rate)).l2(1e-4).list()
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
                .build();

        // run the model
        Constants.model = new MultiLayerNetwork(conf);

        Constants.model.init();
        Constants.model.setListeners(new ScoreIterationListener(5));
        INDArray a = Constants.model.getLayer(0).getParam("W");

        System.out.println("start");


//			tmp1.clear();


        System.out.println("NETWORK2.JAVA is running" + " 784 / " + feature_ratio + "  and " +
                Constants.numberOfNeurons + "  neurons at  " + Constants.output_file_prefix);


        int j = 0;

        while (j != net2.fold_iteration) {
            net2.fivefoldIterator.next();
            j++;
        }


        for (; net2.fold_iteration < 5; net2.fold_iteration++) {


            mnistTrain = net2.fivefoldIterator.next();
            mnistTest = net2.fivefoldIterator.testFold();


            if (deSerializing == false) {
                net2.init_problem_configuration(numInputs, feature_ratio);
                net2.save_kfold(numInputs, feature_ratio, trainSet2, k);
                trainSet2 = _utils.dataset2Instances(mnistTrain);
                saveBatches(Constants.numBatches, Constants.batchSize, mnistTrain, trainSet2);
            } else
                trainSet2 = _utils.dataset2Instances(mnistTrain);


                for (int i = 0 + net2.iteration_based; i < 156; i++) {
                    // in the first iteration do the bagging test and the each batch
                    // test :D
                    for (int b = 0; b < Constants.numBatches; b++) {

                        System.out.println("at fold iteration " + net2.fold_iteration + "   iteration  " + i + "  batch   " + b);

                        DataSet set = getBatchTrainSet(b, Constants.batchSize, trainSet2);

                        if (i % 10 == 0 && serializing && b == Constants.numBatches - 1) {
                            Constants.isSerialzing = true;
                            _utils.serializing();
                            File file = new File(Constants.output_file_prefix + "/problem/problem_configuration");
                            FileWriter fr = new FileWriter(file);
                            BufferedWriter out = new BufferedWriter(fr);
                            String str = new String();
                            str += "iteration_based " + i + "\n";
                            str += "fold " + net2.fold_iteration + "\n";
                            out.write(str);
                            out.close();
                            fr.close();
                        }
                        if (deSerializing == true) {
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
                        System.out.println("EVALUATING AT ITERATION  " + i);
                        Constants.isEvaluating = true;
                        log.info("Evaluate model....");
                        //
                        Evaluation eval = new Evaluation(outputNum); // create an
                        //
    //                while (mnistTest.hasNext()) {

    //                    DataSet next = mnistTest.next();
                        System.out.println(Constants.isEvaluating);
                        _utils.setLabels(mnistTest.getLabels(), Constants.isEvaluating,
                                false);
                        INDArray output = Constants.model.output(mnistTest.getFeatures());

                        eval.eval(mnistTest.getLabels(), output);
    //                }
    //                mnistTest.reset();
                        //
                        String path = Constants.output_file_prefix + "/" + (net2.fold_iteration + 1) + "/resultIteration_" + i;
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

            // run the model
            Constants.model = new MultiLayerNetwork(conf);

            Constants.model.init();
            Constants.model.setListeners(new ScoreIterationListener(5));


        }


    }


    private static void regular_experiment() throws Exception {
        {

            // TODO Nd4j.setDataType(Type.DOUBLE);


            boolean deSerializing = false;
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
            Constants.numberOfNeurons = 50;
            Constants.batchSize = 100;
            Constants.avgHFDepth = new double[Constants.numberOfLayers];
            double numberTrainExamples = 60000d;
            Constants.allData_size = 70000;
            Constants.numBatches = (int) ((numberTrainExamples) / Constants.batchSize);
            Constants.numClasses = 10;
            Constants.maximumDepth = 20;
            Constants.maximumDepth--;
            Constants.output_file_prefix = "/home/research/result/phase4/randomClassConfig/23_pc/23";

            double learning_rate = 0.1;
            int feature_ratio = 2;
            DataSetIterator mnistTrain = null;
            DataSet mnistTest = new DataSet();
            //
            Instances trainSet2 = null;


            Network2 net2 = new Network2();


            if (deSerializing == false) {
                net2.init_problem_configuration(numInputs, feature_ratio);
                trainSet2 = net2.save_problem_configuration(numInputs, feature_ratio, trainSet2);
                DataSet tempTrainSet;
                tempTrainSet = _utils.instancesToDataSet(trainSet2);
                saveBatches(Constants.numBatches, Constants.batchSize, tempTrainSet, trainSet2);
            } else
                trainSet2 = net2.load_problem_configuration(numInputs, feature_ratio);


            FileInputStream mnistTest_file =
                    new FileInputStream(Constants.output_file_prefix + "/problem/mnistTest_file.ser");
            ObjectInputStream mnistTest_file_in = new ObjectInputStream(mnistTest_file);
            mnistTest.load(mnistTest_file);


            mnistTest_file.close();
            mnistTest_file_in.close();


            mnistTrain = null;


            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(6)

                    .trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .weightInit(WeightInit.XAVIER).updater(new Sgd(learning_rate)).l2(1e-4).list()
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
                    .build();

            // run the model
            Constants.model = new MultiLayerNetwork(conf);

            Constants.model.init();
            Constants.model.setListeners(new ScoreIterationListener(5));
            System.out.println("start");


//			tmp1.clear();


            System.out.println("NETWORK2.JAVA is running" + " 784 / " + feature_ratio + "  and " +
                    Constants.numberOfNeurons + "  neurons at  " + Constants.output_file_prefix);
            for (int i = 0 + net2.iteration_based; i < 156; i++) {
                // in the first iteration do the bagging test and the each batch
                // test :D
                for (int b = 0; b < Constants.numBatches; b++) {

                    System.out.println("iteration  " + i + "  batch   " + b);

                    DataSet set = getBatchTrainSet(b, Constants.batchSize, trainSet2);

                    if (i % 10 == 0 && serializing && b == Constants.numBatches - 1) {
                        Constants.isSerialzing = true;
                        _utils.serializing();
                        File file = new File(Constants.output_file_prefix + "/problem/problem_configuration");
                        FileWriter fr = new FileWriter(file);
                        BufferedWriter out = new BufferedWriter(fr);
                        String str = new String();
                        str += "iteration_based " + i + "\n";
                        out.write(str);
                        out.close();
                        fr.close();
                    }
                    if (deSerializing == true) {
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
                    System.out.println("EVALUATING AT ITERATION  " + i);
                    Constants.isEvaluating = true;
                    log.info("Evaluate model....");
                    //
                    Evaluation eval = new Evaluation(outputNum); // create an
                    //
//                while (mnistTest.hasNext()) {

//                    DataSet next = mnistTest.next();
                    System.out.println(Constants.isEvaluating);
                    _utils.setLabels(mnistTest.getLabels(), Constants.isEvaluating,
                            false);
                    INDArray output = Constants.model.output(mnistTest.getFeatures());

                    eval.eval(mnistTest.getLabels(), output);
//                }
//                mnistTest.reset();
                    //
                    String path = Constants.output_file_prefix + "/result/resultIteration_" + i;
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
            int counter = 0;
//        while (mnistTest.hasNext()) {

//            DataSet next = mnistTest.next();
            _utils.setLabels(mnistTest.getLabels(), Constants.isEvaluating, false);
            INDArray output = Constants.model.output(mnistTest.getFeatures());
            eval.eval(mnistTest.getLabels(), output); // check the prediction
            // against the true
            // class
            counter++;
//        }
//			System.out.println(counter);
            log.info(eval.stats());

        }
    }

    private static DataSet getBatchTrainSet(int batchNumber, int batchRate, Instances training) {

//
        DataSet set = null;
        try {
            set = readBatch_dataset(batchNumber);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        int start = batchNumber * batchRate;
        int end = (batchNumber + 1) * batchRate;
        List<Instance> list = training.subList(start, end);

        double[] labels_list = new double[list.size()];
        for (int i = 0; i < list.size(); i++)
            labels_list[i] = list.get(i).classValue();
        Constants.trainInstancesLabel = Nd4j.create(labels_list).transpose();

//        features.cleanup();
//        labels.cleanup();
//        batchTrain_features.cleanup();
//        batchTrain_labels.cleanup();


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


        iteration_based = 0;


    }


    private void save_kfold(int numInputs, int feature_ratio, Instances trainSet2, int k) throws Exception {


        System.out.println("SAVING THE PROBLEM");
        File file = new File(Constants.output_file_prefix + "/problem/problem_configuration");
        FileWriter fr = new FileWriter(file);
        BufferedWriter out = new BufferedWriter(fr);
        String str = new String();
        str += "iteration_based " + iteration_based + "\n";
        str += "fold " + fold_iteration + "\n";
        out.write(str);

        FileOutputStream attributesIndexes_file =
                new FileOutputStream(Constants.output_file_prefix + "/problem/attributesIndexes.ser");
        ObjectOutputStream attIndex_out = new ObjectOutputStream(attributesIndexes_file);
        attIndex_out.writeObject(Constants.attributesIndexes);
//
        FileOutputStream class_file =
                new FileOutputStream(Constants.output_file_prefix + "/problem/class_file.ser");
        ObjectOutputStream class_file_out = new ObjectOutputStream(class_file);
        class_file_out.writeObject(Constants.classChosedArray);
//


//

        if (this.fivefoldIterator == null) {
            DataSetIterator mnistTrain = new MnistDataSetIterator(60000, true, 6);
            DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 6);
            Instances AllInstances = _utils.dataset2Instances(mnistTrain.next());
            Instances temp = _utils.dataset2Instances(mnistTest.next());
            for (int i = 0; i < temp.size(); i++)
                AllInstances.add(temp.get(i));

            DataSet allData = _utils.instancesToDataSet(AllInstances);
            this.fivefoldIterator = new KFoldIterator(5, allData);
            FileOutputStream mnist_kfold =
                    new FileOutputStream(Constants.output_file_prefix + "/problem/mnist_kfold.ser");
            ObjectOutputStream mnist_kfold_out = new ObjectOutputStream(mnist_kfold);
            mnist_kfold_out.writeObject(this.fivefoldIterator);
        }


    }

    private Instances save_problem_configuration(int numInputs, int feature_ratio, Instances trainSet2) throws Exception {

        System.out.println("SAVING THE PROBLEM");
        File file = new File(Constants.output_file_prefix + "/problem/problem_configuration");
        FileWriter fr = new FileWriter(file);
        BufferedWriter out = new BufferedWriter(fr);
        String str = new String();
        str += "iteration_based " + iteration_based + "\n";
        out.write(str);

        FileOutputStream attributesIndexes_file =
                new FileOutputStream(Constants.output_file_prefix + "/problem/attributesIndexes.ser");
        ObjectOutputStream attIndex_out = new ObjectOutputStream(attributesIndexes_file);
        attIndex_out.writeObject(Constants.attributesIndexes);


        FileOutputStream class_file =
                new FileOutputStream(Constants.output_file_prefix + "/problem/class_file.ser");
        ObjectOutputStream class_file_out = new ObjectOutputStream(class_file);
        class_file_out.writeObject(Constants.classChosedArray);

        DataSetIterator mnistTrain = new MnistDataSetIterator(Constants.batchSize, true, 6);
        DataSetIterator mnistTest = new MnistDataSetIterator(10000, false, 6);
        DataNormalization scaler = new NormalizerStandardize();
        scaler.fit(mnistTrain);
        mnistTrain.setPreProcessor(scaler);
        mnistTest.setPreProcessor(scaler); // same normalization for better results
        mnistTrain.reset();


        FileOutputStream mnistTrain_file =
                new FileOutputStream(Constants.output_file_prefix + "/problem/mnistTrain_file.ser");
        ObjectOutputStream mnistTrain_file_out = new ObjectOutputStream(mnistTrain_file);
        mnistTrain_file_out.writeObject(mnistTrain);


        FileOutputStream mnistTest_file =
                new FileOutputStream(Constants.output_file_prefix + "/problem/mnistTest_file.ser");
        ObjectOutputStream mnistTest_file_out = new ObjectOutputStream(mnistTest_file);
//        mnistTest_file_out.writeObject(mnistTest);
        mnistTest.next().save(mnistTest_file);

        int counter = 0;
        Instances trainTemp = null;
        int c = 0;
        while (mnistTrain.hasNext()) {
            DataSet set = mnistTrain.next();
            if (c == 0) {
                trainSet2 = _utils.dataset2Instances(set);
            } else {
                trainTemp = _utils.dataset2Instances(set);
                for (int i = 0; i < trainTemp.size(); i++)
                    trainSet2.add(trainTemp.get(i));
            }

            c++;
        }
        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "" + (trainSet2.numAttributes()); // range of variables to
        convert.setOptions(options);
        convert.setInputFormat(trainSet2);
        trainSet2 = weka.filters.Filter.useFilter(trainSet2, convert);
        trainSet2.setClassIndex(trainSet2.numAttributes() - 1);


        FileOutputStream trainSet_file =
                new FileOutputStream(Constants.output_file_prefix + "/problem/trainSet_file.ser");
        ObjectOutputStream trainSet_file_out = new ObjectOutputStream(trainSet_file);
        trainSet_file_out.writeObject(trainSet2);


        out.close();
        attributesIndexes_file.close();
        attIndex_out.close();
        class_file.close();
        class_file_out.close();
        mnistTrain_file.close();
//        mnistTrain_file_out.close();
        mnistTrain_file.close();
//        mnistTrain_file_out.close();
        mnistTest_file.close();
        mnistTest_file_out.close();
        trainSet_file_out.close();
        trainSet_file.close();
        return trainSet2;
    }

    private void load_kfold_configuration() throws IOException, ClassNotFoundException {

        System.out.println("LOADING THE PROBLEM");
        File file = new File(Constants.output_file_prefix + "/problem/problem_configuration");

        Scanner in = new Scanner(file);

        String str = in.nextLine();
        String[] arr = str.split(" ");
        this.iteration_based = Integer.parseInt(arr[1]);

        str = in.nextLine();
        arr = str.split(" ");
        this.fold_iteration = Integer.parseInt(arr[1]);


        FileInputStream attributesIndexes_file =
                new FileInputStream(Constants.output_file_prefix + "/problem/attributesIndexes.ser");
        ObjectInputStream attIndex_in = new ObjectInputStream(attributesIndexes_file);

        Constants.attributesIndexes = (HashMap<Integer, int[]>) attIndex_in.readObject();


        FileInputStream class_file =
                new FileInputStream(Constants.output_file_prefix + "/problem/class_file.ser");
        ObjectInputStream class_file_in = new ObjectInputStream(class_file);

        Constants.setClassChosedArray((HashMap<Integer, ArrayList<Integer>>) class_file_in.readObject());


        FileInputStream mnist_kfold =
                new FileInputStream(Constants.output_file_prefix + "/problem/mnist_kfold.ser");
        ObjectInputStream mnist_kfold_in = new ObjectInputStream(mnist_kfold);

        this.fivefoldIterator = (KFoldIterator) mnist_kfold_in.readObject();


//        FileInputStream mnistTrain_file =
//                new FileInputStream("../problem/mnistTrain_file.ser");
//        ObjectInputStream mnistTrain_file_in = new ObjectInputStream(mnistTrain_file);
//        mnistTrain = (DataSetIterator) mnistTrain_file_in.readObject();
//
//
//        FileInputStream mnistTest_file =
//                new FileInputStream("../problem/mnistTest_file.ser");
//        ObjectInputStream mnistTest_file_in = new ObjectInputStream(mnistTest_file);
//        mnistTest = (DataSetIterator) mnistTest_file_in.readObject();


//        mnistTest_file.close();
//        mnistTest_file_in.close();
//        mnistTrain_file.close();
//        mnistTrain_file_in.close();
        class_file.close();
        class_file_in.close();
        attributesIndexes_file.close();
        attIndex_in.close();
        in.close();


    }

    private Instances load_problem_configuration(int numInputs, int feature_ratio) throws IOException, ClassNotFoundException {

        System.out.println("LOADING THE PROBLEM");
        File file = new File(Constants.output_file_prefix + "/problem/problem_configuration");

        Scanner in = new Scanner(file);

        String str = in.nextLine();
        String[] arr = str.split(" ");
        this.iteration_based = Integer.parseInt(arr[1]);

        FileInputStream attributesIndexes_file =
                new FileInputStream(Constants.output_file_prefix + "/problem/attributesIndexes.ser");
        ObjectInputStream attIndex_in = new ObjectInputStream(attributesIndexes_file);

        Constants.attributesIndexes = (HashMap<Integer, int[]>) attIndex_in.readObject();


        FileInputStream class_file =
                new FileInputStream(Constants.output_file_prefix + "/problem/class_file.ser");
        ObjectInputStream class_file_in = new ObjectInputStream(class_file);

        Constants.setClassChosedArray((HashMap<Integer, ArrayList<Integer>>) class_file_in.readObject());


        FileInputStream trainSet_file =
                new FileInputStream(Constants.output_file_prefix + "/problem/trainSet_file.ser");
        ObjectInputStream trainSet_file_in = new ObjectInputStream(trainSet_file);
        Instances trainTest2 = (Instances) trainSet_file_in.readObject();


//        FileInputStream mnistTrain_file =
//                new FileInputStream("../problem/mnistTrain_file.ser");
//        ObjectInputStream mnistTrain_file_in = new ObjectInputStream(mnistTrain_file);
//        mnistTrain = (DataSetIterator) mnistTrain_file_in.readObject();
//
//
//        FileInputStream mnistTest_file =
//                new FileInputStream("../problem/mnistTest_file.ser");
//        ObjectInputStream mnistTest_file_in = new ObjectInputStream(mnistTest_file);
//        mnistTest = (DataSetIterator) mnistTest_file_in.readObject();


//        mnistTest_file.close();
//        mnistTest_file_in.close();
//        mnistTrain_file.close();
//        mnistTrain_file_in.close();
        class_file.close();
        class_file_in.close();
        attributesIndexes_file.close();
        attIndex_in.close();
        in.close();
        trainSet_file.close();
        trainSet_file_in.close();
        return trainTest2;


    }


    private static void saveBatches(int totalBatches, int batchRate, DataSet trainSet, Instances training) throws IOException {

        for (int batchNumber = 0; batchNumber < totalBatches; batchNumber++) {


            INDArray features = trainSet.getFeatures();
            INDArray labels = trainSet.getLabels();
            int start = batchNumber * batchRate;
            int end = (batchNumber + 1) * batchRate;

            INDArray batchTrain_features = features.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());
            INDArray batchTrain_labels = labels.get(NDArrayIndex.interval(start, end), NDArrayIndex.all());

            DataSet set = new DataSet(batchTrain_features, batchTrain_labels);
//b/batch" + batchNumber + ".ser");
            FileOutputStream batch_file =
                    new FileOutputStream("batch" + batchNumber + ".ser");
            ObjectOutputStream batch_file_out = new ObjectOutputStream(batch_file);
            batch_file_out.writeObject(set);


            batch_file.close();
            batch_file_out.close();
//            instanc_b_file.close();
//            instanc_b_file_out.close();
        }
    }


    private static DataSet readBatch_dataset(int batchNumber) throws IOException, ClassNotFoundException {


//        FileInputStream batch_file =
//                new FileInputStream(Constants.output_file_prefix + "/data/batch" + batchNumber + ".ser");
        FileInputStream batch_file =
                new FileInputStream("batch" + batchNumber + ".ser");
        ObjectInputStream batch_file_in = new ObjectInputStream(batch_file);
        DataSet set = (DataSet) batch_file_in.readObject();

        batch_file.close();
        batch_file_in.close();
        return set;
    }
}


