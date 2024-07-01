<template>
	<div class="container my-5">
		<h2 class="font-weight-bold">Stochastic Linear Regression (Tensorflow.js)</h2>
		<p class="mt-4 mb-5">
			The following is a Stochastic Linear Regression implementation using Tensorflow.js,
			running against a house pricing dataset.

			Your machine is probably now currently training the model as you are reading this.
			It will automatically stop training after reaching some optimal values.

			<br>
		</p>

		<apexchart height="480" type="line" :options="chartOptions" :series="chartSeries"></apexchart>

		<div class="row no-gutters prediction-form mt-2">
			<div class="col-auto d-flex flex-column justify-content-center">
				A
			</div>
			<div class="col-auto mx-2">
				<input type="number" class="form-control form-control-sm bg-transparent" v-model="formArea">
			</div>
			<div class="col d-flex flex-column justify-content-center">
				<span>
					sqft house is going to likely be priced: <strong>{{ prediction ? prediction.toLocaleString(undefined, {maximumFractionDigits: 0}) : 0 }} US$</strong>
				</span>
			</div>
		</div>

		<table class="table borderless w-100 mt-5">
			<tr>
				<th>R<sup>2</sup>:</th>
				<td>{{ R2.toFixed(3) }}</td>

				<th>Mean Squared Error:</th>
				<td>
					{{ mse.toLocaleString() }}
					<span class="text-success">
						({{ mseHistory.length >= 2 ? (mse - mseHistory[1]).toLocaleString() : 0 }})
					</span>
				</td>

			</tr>
			<tr>
				<th>Iterations:</th>
				<td>{{ iterations }}</td>

				<th>Iteration cost (avg.):</th>
				<td>{{ (trainingTimes.reduce((acc, val) => acc + val, 0) / trainingTimes.length || 1).toFixed(0) }}ms</td>
			</tr>
			<tr>
				<th>Dataset size*:</th>
				<td>150 (training) / 300 (testing)</td>

				<th>Batch size:</th>
				<td>{{ batchSize }}</td>
			</tr>
		</table>

		<p class="text-muted mt-3">
			<small>
				* Testing set is larger than training set in order to keep each training iteration faster.
			</small>
		</p>
	</div>
</template>

<script>
	import VueApexCharts from 'vue-apexcharts';
	import { io } from "socket.io-client";

	export default {
		components: {
			'apexchart': VueApexCharts,
		},

		mounted() {
			this.socket = io('localhost:5000');

			this.socket.on('data', (...args) => {
				this.features = args[0].features;
				this.labels = args[0].labels;
			});

			this.socket.on('update', (...args) => {
				this.R2 = args[0].r2;
				this.mse = args[0].mse;
				this.iterations = args[0].iterations;
				this.mseHistory = args[0].mseHistory;
				this.trainingTimes = args[0].trainingTimes;
				this.predictions = args[0].predictions;
			});

			this.socket.on('prediction', (...args) => {
				this.prediction = args[0].price;
			});
		},
		
		data() {
			return {
				socket: null,
				
				batchSize: 50,

				formArea: 500,

				features: [],
				labels: [],
				predictions: [],

				testingFeatures: [],
				testingLabels: [],

				regression: null,
				prediction: null,

				R2: 0,
				mse: 0,
				iterations: 0,
				trainingTimes: [],
				mseHistory: [],

				chartOptions: {
					chart: {
						animations: {
							enabled: false,
						},
					},

					markers: {
						size: [6, 0]
					},

					tooltip: {
						shared: false,
						intersect: true,
					},

					xaxis: {
						min: 0,

						title: {
							text: 'Living area (sqft)',
						},
					},

					yaxis: {
						min: 0,

						title: {
							text: 'Price (US$)',
						},
					}
				},
			};
		},

		watch: {
			formArea() {
				if (this.socket)
					this.socket.emit('predict', {
						'area': this.formArea
					})
			}
		},

		computed: {
			chartSeries() {
				const houses = [];
				const predictions = [];
				
				for (let i = 0; i < this.features.length; i++) {
					houses.push({x: this.features[i], y: this.labels[i]});
					
					if (this.predictions.length > 0)
						predictions.push({x: this.features[i], y: this.predictions[i]});
				}

				return [
					{
						name: 'Data points',
						type: 'scatter',
						data: houses,
					},
					{
						name: 'Predictions',
						type: 'line',
						data: predictions,
					}
				];
			},
		}
	};
</script>


<style scoped>
	table {
		table-layout: fixed;
	}

	.prediction-form input {
		width: 96px;
	}
</style>
