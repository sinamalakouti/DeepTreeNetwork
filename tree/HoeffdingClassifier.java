package tree;

import java.util.ArrayList;
import java.util.Enumeration;
import weka.classifiers.trees.J48;
import weka.core.*;

public class HoeffdingClassifier  {

	int [] stat;

	int n = 0; //TODO




	public void buildTree(Instances S, ArrayList<Integer> attributes, double delta) {

		Node root = new Node(-1, -1, null, true);
		
	    Enumeration instEnum = S.enumerateInstances();
	    stat = new int[S.numClasses()];
	    
	    
	    while (instEnum.hasMoreElements()) {
	    	
	    	
	    	
	    	
	    }

		


	}
//	
//	private void sort(Attribute att , int classId) {
//		this.stat[classId] ++;
//		this.n ++;
//		if () {
//			
//		}
		
		
//	}
	

//    def sortExample(self,attribValue,classId):
//        self.stat[classId] += 1
//        self.n += 1
//        if attribValue == 0:
//            if self.left == None:
//                newLeaf = Hleaf()
//                #newLeaf.setLeafId(attribId,attribValue)
//                newLeaf.sortExample(classId)
//                self.left = newLeaf
//            else:
//                self.left.sortExample(classId)
//        elif attribValue == 1:
//            if self.right == None:
//                newLeaf = Hleaf()
//                #newLeaf.setLeafId(attribId,attribValue)
//                newLeaf.sortExample(classId)
//                self.right = newLeaf
//            else:
//                self.right.sortExample(classId)
                
                
                

	public double gain(Instances data , Attribute att) {
		
		double infoGain = shannon_entropy(data);
		
		Instances[] splitData = this.splitData(data, att);
		for (int j = 0; j < att.numValues(); j++) {
			
			if (splitData[j].numInstances() > 0 ) {
				
				infoGain -= ((double) splitData[j].numInstances() /
					     (double) data.numInstances()) *
				  shannon_entropy(splitData[j]);
			}
		}
			
		
		return infoGain;
	}

	public double shannon_entropy(Instances data ) {

		double entropy = 0d;
		double n = data.size();
		int n_class = data.numClasses();

		double [] classCounts = new double[n_class];

		Enumeration instEnum = data.enumerateInstances();

		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()] ++;
		}

		for (int j = 0; j < data.numClasses(); j++) {
			if (classCounts[j] > 0 ) {
				double p = classCounts[j]/ n;
				entropy -=  p * Utils.log2(p);
			}
		}


		return entropy;
	}
	
	  private Instances[] splitData(Instances data, Attribute att) {

		    Instances[] splitData = new Instances[att.numValues()];
		    for (int j = 0; j < att.numValues(); j++) {
		      splitData[j] = new Instances(data, data.numInstances());
		    }
		    Enumeration instEnum = data.enumerateInstances();
		    while (instEnum.hasMoreElements()) {
		      Instance inst = (Instance) instEnum.nextElement();
		      splitData[(int) inst.value(att)].add(inst);
		    }
		    return splitData;
		  }
}
