package org.example.wineprediction;

// import com.amazonaws.auth.AWSCredentials;
// import com.amazonaws.auth.EnvironmentVariableCredentialsProvider;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;

public class App {

        public static final Logger logger = LogManager.getLogger(App.class);

        private static final String ACCESS_KEY_ID = "ASIA5FTY7O2RKQ5H7DC4"; // System.getProperty("ACCESS_KEY_ID");
        private static final String SECRET_KEY = "Da1IoFrBDNAP1ZI4ewftGnG1FPsbFuk/57ebEfCI"; // System.getProperty("SECRET_KEY");

        private static final String MASTER_URI = "local[*]";

        public static void main(String[] args) {

                Logger.getLogger("org").setLevel(Level.ERROR);
                Logger.getLogger("akka").setLevel(Level.ERROR);
                Logger.getLogger("breeze.optimize").setLevel(Level.ERROR);
                Logger.getLogger("com.amazonaws.auth").setLevel(Level.DEBUG);

                SparkSession spark = SparkSession.builder()
                                .appName("Wine Quality Prediction App").master(MASTER_URI)
                                .config("spark.executor.memory", "3g")
                                .config("spark.driver.memory", "3g")
                                .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.2")
                                .getOrCreate();

                spark.sparkContext().hadoopConfiguration().set("fs.s3a.aws.credentials.provider",
                                "com.amazonaws.auth.InstanceProfileCredentialsProvider,com.amazonaws.auth.DefaultAWSCredentialsProviderChain");
                spark.sparkContext().hadoopConfiguration().set("fs.s3a.access.key", ACCESS_KEY_ID);
                spark.sparkContext().hadoopConfiguration().set("fs.s3a.secret.key", SECRET_KEY);

                LogisticRegressionV2 parser = new LogisticRegressionV2();
                parser.trainModel(spark);

                spark.stop();
        }
}
