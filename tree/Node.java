package tree;

import java.util.ArrayList;

public class Node {

	
	private int edge ; 
	private int feature;
	private ArrayList<Integer> labels ;
	private boolean isLeaf ;
	
	
	public Node (int edge, int feature,  ArrayList<Integer> labels, boolean isLeaF) {
		
		this.edge = edge;
		this.feature = feature;
		this.labels = labels;
		this.isLeaf = isLeaF;
	}


	public int getEdge() {
		return edge;
	}


	public void setEdge(int edge) {
		this.edge = edge;
	}


	public int getFeature() {
		return feature;
	}


	public void setFeature(int feature) {
		this.feature = feature;
	}


	public ArrayList<Integer> getLabels() {
		return labels;
	}


	public void setLabels(ArrayList<Integer> labels) {
		this.labels = labels;
	}


	public boolean isLeaf() {
		return isLeaf;
	}


	public void setLeaf(boolean isLeaf) {
		this.isLeaf = isLeaf;
	}
	
	
	
}
