package com.fajaralmu.neuralnetwork;

import java.util.Arrays;

import com.fajaralmu.neuralnetwork.train.TrainSet;

public class App {

	// 1 - size, 2 - neuron
	private double[][] output;
	private double[][][] weight;
	private double[][] bias;

	private double[][] errorSignal;
	private double[][] outputDerivative;

	public final int[] NETWORK_LAYER_SIZES;
	public final int INPUT_SIZE;
	public final int OUTPUT_SIZE;
	public final int NETWORK_SIZE;

	public App(int... layerSize) {

		this.NETWORK_LAYER_SIZES = layerSize;
		this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
		this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
		this.OUTPUT_SIZE = this.NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];

		this.output = new double[NETWORK_SIZE][];
		this.weight = new double[NETWORK_SIZE][][];
		this.bias = new double[NETWORK_SIZE][];

		this.errorSignal = new double[NETWORK_SIZE][];
		this.outputDerivative = new double[NETWORK_SIZE][];
		/**
		 * setting the array sizes
		 */
		for (int i = 0; i < NETWORK_SIZE; i++) {
			this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
			this.bias[i] = new double[NETWORK_LAYER_SIZES[i]];
			this.errorSignal[i] = new double[NETWORK_LAYER_SIZES[i]];
			this.outputDerivative[i] = new double[NETWORK_LAYER_SIZES[i]];

			if (i > 0) { // the 0 layers has no weight
				weight[i] = new double[NETWORK_LAYER_SIZES[i]][NETWORK_LAYER_SIZES[i - 1]];
			}
		}
	}

	public double[] calculate(double... input) {
		if (input.length != this.INPUT_SIZE) {
			return null;
		}

		/**
		 * first layer is not layer but it is buffer
		 */
		this.output[0] = input;

		for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {

			for (int neuron = 0; neuron < this.NETWORK_LAYER_SIZES[layer]; neuron++) {

				double sum = bias[layer][neuron];

				for (int prevNeuron = 0; prevNeuron < this.NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {

					sum += output[layer - 1][prevNeuron] * this.weight[layer][neuron][prevNeuron];
				}

				output[layer][neuron] = sigmoid(sum);
				outputDerivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
			}
		}

		return output[this.NETWORK_SIZE - 1]; // last index
	}

	public void train(double[] input, double[] target, double learningRate) {
		if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
			return;
		}

		calculate(input);
		backPropError(target);
		updateWeight(learningRate);
	}

	public void backPropError(double[] target) {
		for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {

			errorSignal[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron])
					* outputDerivative[NETWORK_SIZE - 1][neuron];
		}

		for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
			for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
				double sum = 0;
				for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
					sum += weight[layer + 1][nextNeuron][neuron] * errorSignal[layer + 1][nextNeuron];
				}
				this.errorSignal[layer][neuron] = sum * this.outputDerivative[layer][neuron];
			}
		}
	}

	public void updateWeight(double learningRate) {

		for (int layer = 1; layer < NETWORK_SIZE; layer++) {
			for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
				double delta = -learningRate * errorSignal[layer][neuron];
				bias[layer][neuron] += delta;
				for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
					weight[layer][neuron][prevNeuron] += delta * output[layer - 1][prevNeuron];
				}

			}
		}
	}

	private double sigmoid(double x) {
		return 1d / (1 + Math.exp(-x));
	}

	public void train(TrainSet trainSet, int loops, int batchSize) {
		for (int i = 0; i < loops; i++) {
			TrainSet batchExtracted = trainSet.extractBatch(batchSize);
			for (int batch = 0; batch < batchSize; batch++) {
				double learningRate = 0.3;
				this.train(batchExtracted.getInput (batch), batchExtracted.getOutput(batch), learningRate );
				
			}
		}
	}

	public static void mainOld(String[] args) {
		// TODO Auto-generated method stub
		App app = new App(4, 1, 3, 4);

		double[] input = new double[] { 0.1, 0.5, 0.2, 0.9 };
		double[] target = new double[] { 0, 1, 0, 0 };

		double[] input2 = new double[] { 0.6, 0.1, 0.4, 0.8 };
		double[] target2 = new double[] { 0, 1, 0, 0 };

		/**
		 * learning
		 */
		for (int i = 0; i < 100000; i++) {
			app.train(input, target, 0.3);
		}

		/**
		 * testing
		 */
		double[] o = app.calculate(input);
		System.out.println(Arrays.toString(o));

	}
	
	static double[] getDouble(double... doubles) { 
		
		return doubles;
	}
	
	public static void main(String[] args) {
		
		App app = new App(4,3,3,2);
		
		TrainSet trainSet = new TrainSet(4, 2);
		trainSet.addData(getDouble(0.1,0.2,0.3,0.4) , getDouble(0.9, 0.1));
		trainSet.addData(getDouble(0.9,0.8,0.7,0.6) , getDouble(0.1, 0.9)); 
		trainSet.addData(getDouble(0.3,0.8,0.1,0.4) , getDouble(0.3, 0.7)); 
		trainSet.addData(getDouble(0.9,0.8,0.1,0.2) , getDouble(0.7, 0.3));
		app.train(trainSet, 100000, 4);
		
		for (int i = 0; i < 4; i++) {
			System.out.println(Arrays.toString(app.calculate(trainSet.getInput(i))));
		}

	}

}
