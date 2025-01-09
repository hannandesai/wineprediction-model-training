# wineprediction-model-training

This repository contains the application for training a Machine Learning model to predict wine quality. The application uses Apache Spark on an AWS EMR Cluster and is implemented in Java using the Maven build tool.

## Features

- **Model Training**: Trains a Linear Regression model for wine quality prediction.
- **Cloud Integration**: Executes on AWS EMR Cluster across multiple EC2 instances.
- **Storage**: Saves the trained model in an Amazon S3 bucket for future use.

## Prerequisites

- Java Development Kit (JDK)
- Maven
- AWS Account
- Apache Spark setup on AWS EMR Cluster
- Training and validation datasets (`TrainingDataSet.csv` and `ValidationDataSet.csv`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hannandesai/wineprediction-model-training.git
   cd wineprediction-model-training
   ```
2. Build the project:
   ```bash
   mvn clean package
   ```

## Usage

1. Connect to the AWS EMR cluster.
2. Upload the training and validation datasets to an S3 bucket.
3. Create a JAR file:
   ```bash
   mvn clean package
   ```
   The JAR file will be created in the `target` folder as `wineprediction-1.0-SNAPSHOT-jar-with-dependencies.jar`.
4. Upload the JAR file to the S3 bucket.
5. Submit the Spark job:
   ```bash
   spark-submit s3://<bucket-name>/wineprediction-1.0-SNAPSHOT-jar-with-dependencies.jar
   ```
6. Monitor the job using the Spark History Server UI.
7. The trained model, named `LinearRegressionModel`, will be saved in the S3 bucket.
