# House Price Prediction using Stochastic Linear Regression

This repository contains a data mining project designed to predict house prices using Stochastic Linear Regression. The project is split into two main parts:
1. **Python Backend**: This part handles data mining and implements the Stochastic Linear Regression algorithm.
2. **Vue.js Frontend**: This part is a web application that provides an interactive interface for visualizing the data and predictions in real-time.

## Table of Contents

- [Overview](#overview)
  - [What is SLR](#what-is-slr)
- [Setup and Installation](#setup-and-installation)
  - [Python Backend](#python-backend)
  - [Vue.js Frontend](#vuejs-frontend)
- [Usage](#usage)

## Overview

The objective of this project is to predict house prices based on various features such as size, location, number of bedrooms, etc., using Stochastic Linear Regression. The Python backend is responsible for data processing, training the model, and serving predictions via an API. The Vue.js frontend interacts with the backend to display the predictions and allow for user interactivity.

### What is SLR?
- Linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables).
- There are many approaches to implementing linear regression, most notably using gradient descent.
- Gradient descent is an optimization function that is mostly used to update weights in training functions, using the observed data and an error function.
- Stochastic linear regression (SLR) uses a notion called Stochastic gradient descent (SGD) rather than a normal gradient descent.
- The only difference is that, while the normal gradient descent uses all of the data to update the weights, SGD uses only a subset of the data.
- This is mostly efficient when the training samples being used are very large, hence needing significant computation power to calculate the gradient descent in every iteration.
- SGD often converges much faster compared to GD but the error function is not as well minimized as in the case of GD. Often in most cases, the close approximation that you get in SGD for the parameter values are enough because they reach the optimal values and keep oscillating there.


## Setup and Installation

### Python Backend

1. **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the backend application:**
    ```bash
    python Main.py
    ```
    This will start the backend server, typically accessible at `http://localhost:65432` through a Socket.IO connection.

### Vue.js Frontend

1. **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2. **Install the dependencies:**
    ```bash
    npm install
    ```

3. **Run the frontend application:**
    ```bash
    npm run serve
    ```
    This will start the frontend development server, typically accessible at `http://localhost:8080`.

## Usage

1. Ensure the backend server is running.
2. Start the frontend development server.
3. Open a web browser and navigate to `http://localhost:8080`.
4. Interact with the web application to view house price predictions in real-time.
