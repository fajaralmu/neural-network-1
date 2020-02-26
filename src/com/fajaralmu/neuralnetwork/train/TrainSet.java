package com.fajaralmu.neuralnetwork.train;

import java.util.ArrayList;

public class TrainSet {

	public final int INPUT_SIZE;
	public final int OUTPUT_SIZE;

	private ArrayList<double[][]> data = new ArrayList<>();

	public TrainSet(int inputSize, int outputSize) {
		this.INPUT_SIZE = inputSize;
		this.OUTPUT_SIZE = outputSize;
	}

	public void addData(double[] in, double[] expected) {
		if (in.length != INPUT_SIZE || expected.length != OUTPUT_SIZE)
			return;

		data.add(new double[][] { in, expected });
	}

	public double[] getOutput(int index) {
		if (index >= 0 && index < size()) {
			return data.get(index)[1];
		}

		return null;
	}
	
	public double[] getInput(int index) {
		if (index >= 0 && index < size()) {
			return data.get(index)[0];
		}

		return null;
	}

	public TrainSet extractBatch(int size) {
		return this;
	}

	public int size() {
		return data.size();
	}

}
