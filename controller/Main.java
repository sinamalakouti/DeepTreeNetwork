package controller;

import java.awt.BorderLayout;
import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import play.mvc.WebSocket.In;
import utils._utils;
import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.j48.NBTreeClassifierTree;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class Main {
	
	
	public static void main(String[] args) throws Exception {
		 
		
		
		
		
		 RecordReader recordReader = new CSVRecordReader();
		recordReader.initialize(new FileSplit(new File("iris.txt")));

		CSVLoader loader = new CSVLoader();
		loader.setSource(new File("iris.txt"));
		String[] options = new String[1];
		options[0] = "-H";
		loader.setOptions(options);

		Instances dataset = loader.getDataSet();
		
		NumericToNominal convert = new NumericToNominal();
		options = new String[2];
		options[0] = "-R";
		options[1] = "" + (dataset.numAttributes()); // range of variables to make numeric
		convert.setOptions(options);
		convert.setInputFormat(dataset);
		dataset = weka.filters.Filter.useFilter(dataset, convert);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		// Second: the RecordReaderDataSetIterator handles conversion to DataSet
		// objects, ready for use in neural network
		int labelIndex = 4; // 5 values in each row of the iris.txt CSV: 4 input features followed by an
		// integer label (class) index. Labels are the 5th value (index 4) in each row
		int numClasses = 3; // 3 classes (types of iris flowers) in the iris data set. Classes have integer
		// values 0, 1 or 2
		int batchSize = 150; // Iris data set: 150 examples total. We are loading all of them into one
		// DataSet (not recommended for large data sets)

		DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
		DataSet allData = iterator.next();
		allData.shuffle();
		SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65); // Use 65% of data for training

		DataSet trainingData = testAndTrain.getTrain();
		DataSet testData = testAndTrain.getTest(); 
//		
		java.util.Random rand = new java.util.Random(3);	
		Instances randData = new Instances(dataset);
		randData.setClassIndex(randData.numAttributes() - 1);
//		randData.randomize(rand);
//		randData.stratify(10);
//
		Instances training = randData.trainCV(10, 0);
		training.setClassIndex(training.numAttributes() - 1);
		Instances test = randData.testCV(10, 0);
		test.setClassIndex(test.numAttributes() - 1);
		training.setClassIndex(training.numAttributes() - 1);
		test.setClassIndex(test.numAttributes() - 1);
		trainingData = _utils.instancesToDataSet(training);
		testData = _utils.instancesToDataSet(test);
		
		
		
//		J48 tree = new J48();
////		tree.buildClassifier(training);
//		System.out.println(training.size());
//		System.out.println(tree.m_root.getMu()[0][0]);
//		System.out.println(tree.m_root.getMu()[0][1]);
//		System.out.println(tree.m_root.getMu()[0][2]);
//		// display classifier
//	     final javax.swing.JFrame jf = 
//	       new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
//	     jf.setSize(500,400);
//	     jf.getContentPane().setLayout(new BorderLayout());
//	     TreeVisualizer tv = new TreeVisualizer(null,
//	         tree.graph(),
//	         new PlaceNode2());
//	     jf.getContentPane().add(tv, BorderLayout.CENTER);
//	     jf.addWindowListener(new java.awt.event.WindowAdapter() {
//	       public void windowClosing(java.awt.event.WindowEvent e) {
//	         jf.dispose();
//	       }
//	     });
// 
//	     jf.setVisible(true);
//	     tv.fitToScreen();
	     
//	     	Matrix meanMatrix  = new Matrix(sampleMeans);    
		NaiveBayes bayes = new NaiveBayes();
		bayes.buildClassifier(training);
		Instances data = new Instances(training);
		double [][] treeAnswers  = new double[10][3];
		double [][] bayesAnswers  = new double[10][3];

		 double[][] mu;
		 double[][] sd;
		 double[] classProb;

			Instances[] localInstances;
			mu = new double[data.numAttributes() - 1][data.numClasses()];
			sd = new double[data.numAttributes() - 1][data.numClasses()];
			classProb = new double[data.numClasses()];
			Instances[] tempInstances = new Instances[data.numClasses()];
		


			for (int c = 0; c < data.numClasses(); c++) {
				RemoveWithValues rwv = new RemoveWithValues();
				String[] options1 = new String[5];
				options1[0] = "-C";
				options1[1] = "" + (data.classIndex() + 1);
				options1[2] = "-L";
				options1[3] = "" + (c + 1);
				options1[4] = "-V";
				rwv.setOptions(options1);
				rwv.setInputFormat(data);
				Instances xt = Filter.useFilter(data, rwv);
				tempInstances[c] = xt;
				classProb[c] = ( (double) xt.size()) / ((double) data.size());
			}
			for (int i = 0; i < data.numAttributes(); i++) {
				if (!data.attribute(i).equals(data.classAttribute())) {

					for (int c = 0; c < data.numClasses(); c++) {

						if (tempInstances[c].size() == 0) {
							mu[i][c] = 0;
							mu[i][c] = 0;
						} else {
							
							double mm = 0d;
							int counter =0;
							for ( int l =0 ; l < data.size() ; l ++ ) {
								
								if ( data.get(l).classValue() == 0) {
									counter ++;
									mm += data.get(0).value(data.attribute(3));
								}
								
								
							}
							
							mm = mm / counter;
							mu[i][c] = tempInstances[c].attributeStats(i).numericStats.mean;
							sd[i][c] = tempInstances[c].attributeStats(i).numericStats.stdDev;
						}
//	        		mu.add(  data.attributeStats(i).numericStats.mean);
//	        		sd.add((data.attributeStats(i).numericStats.stdDev));

					}

				}

			}
//		double []  a = tree.predicate(test.firstInstance());
/*		double [] []b = getGaussianPDF(test.firstInstance(),classProb,mu,sd);
*/		double c = bayes.classifyInstance(test.get(0));
		System.out.println();
		for ( int i =0 ; i < 10 ; i ++) {
			
//			treeAnswers[i] = tree.distributionForInstance(test.get(i));
			bayesAnswers[i] = bayes.distributionForInstance(test.get(i));
		}
		Evaluation eval = new  Evaluation(test);
	eval.evaluateModel(bayes, test);
		int size = 784 / 40;
		INDArray nd = null;
		int a [];
		ArrayList<Integer> arr = new ArrayList<>();
		
		    int[] ret = new int[arr.size()];
		    Iterator<Integer> iter = arr.iterator();
		    for (int i = 0; i < ret.length; i++)
		    {
		     
		        ret[i] = iter.next().intValue();
		    }
	
		nd.getColumns(ret);
		for ( int i = 0 ; i < training.numAttributes() -1 ; i++) {
			arr.add(i);
		}
		int max = 784 / 40;
		HashMap<Integer, int[]> attInexes = new HashMap<>();
		for( int j = 0 ; j < 40 ; j++) {
			Collections.shuffle(arr);
			int [] temp = new int [max];
			for (int i = 0 ; i < max ; i ++) {
				temp[i]=arr.get(0);
			}
			
			attInexes.put(j, temp);
			
		}
		for ( int i = 0 ; i<  10; i ++ ) {
			
			System.out.println("instance number " + i );
//			System.out.println("tree: \t" +   treeAnswers[i][0] + "   " +  treeAnswers[i][1] +  "   " + treeAnswers[i][2]);
			System.out.println("bayes: \t" +   bayesAnswers[i][0] + "   " +  bayesAnswers[i][1] +  "   " + bayesAnswers[i][2]);

		}

//		final javax.swing.JFrame jf = 
//			       new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
//			     jf.setSize(500,400);
//			     jf.getContentPane().setLayout(new BorderLayout());
//			     TreeVisualizer tv = new TreeVisualizer(null,
//			         bayes.graph(),
//			         new PlaceNode2());
//			     jf.getContentPane().add(tv, BorderLayout.CENTER);
//			     jf.addWindowListener(new java.awt.event.WindowAdapter() {
//			       public void windowClosing(java.awt.event.WindowEvent e) {
//			         jf.dispose();
//			       }
//			     });
//
//			     jf.setVisible(true);
//			     tv.fitToScreen();
	
	
	}
	
	  public static  double [][] getGaussianPDF(Instance data , double [] classProb , double[][] sampleMeans , double[][] sampleStDevs ) {
			 ArrayList<Double> [] pdf = new ArrayList [data.numClasses()];
			  double [][] result = new double[1][2];
			  double denominator = 0d;
			  if ( sampleMeans.length != sampleStDevs.length || sampleMeans[0].length != sampleStDevs[0].length)
			  {
				  System.out.println("ERRRROOOOORRRR !!");
				  System.exit(0);
			  }
			  
			  int n_attributes = sampleMeans.length;
			  int n_classes = sampleMeans[0].length;

			  for ( int index  = 0 ; index < 1 ; index ++) {
				  
				  
				  double max = Double.MIN_VALUE;
				  double maxIndex = -1;
				  Instance x = data;
				  for ( int c = 0 ; c < n_classes ; c ++) {
					  double res = 1d;
					  for ( int i = 0 ; i < n_attributes ; i++) {
						  
						  Attribute attribute = data.attribute(i);
						  double power = Math.pow((x.value(attribute)  - sampleMeans[i][c]), 2) / (2 * Math.pow(sampleStDevs[i][c], 2));
						  double temp =  ( 1 /  ( Math.sqrt(2 *  Math.PI ) * sampleStDevs[i][c])) * Math.exp( -1  *  power);
						  res = res * temp;
						  
					  }
					  res = res * classProb[c];
					  denominator += res;
					  if ( res > max) {
						  max = res;
						  maxIndex = c;
					  }
					///  pdf[c].add(res);
					  
					  
				  }
				  
				  result[index][0] = max  /  denominator;
				  result[index][1]= maxIndex;
			  }
			  
			  
			  
			  return result;
			  
		  }

}
