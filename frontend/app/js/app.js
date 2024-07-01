import Vue from 'vue';
import App from './components/App';
import VueApexCharts from 'vue-apexcharts';
import PaceProgress from 'pace-progress';

Vue.use(VueApexCharts);

const app = new Vue({
	template: '<App/>',
	components: {App},
});

app.$mount('#app');

PaceProgress.start();