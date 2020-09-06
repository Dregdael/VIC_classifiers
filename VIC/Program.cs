using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.Collections.Generic;
using System.Data;
using Microsoft.ML.Data;
using System.Linq;

namespace VIC
{
    class Program
    {

        public class ModelOutput
        {
            // ColumnName attribute is used to change the column name from
            // its default value, which is the name of the field.
            [ColumnName("PredictedLabel")]
            public String Prediction { get; set; }
            public float[] Score { get; set; }
            public float[] Probability { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }


        public class BasicEvaluation
        {
            public int TP = 0;
            public int TN = 0;
            public int FN = 0;
            public int FP = 0;
        }

        public static double ComputeTwoClassAUC(BasicEvaluation basicEvaluation)
        {
            double positives = basicEvaluation.TP + basicEvaluation.FN;
            double negatives = basicEvaluation.TN + basicEvaluation.FP;
            var tprate = positives > 0.0 ? basicEvaluation.TP / positives : 1.0;
            var fprate = negatives > 0.0 ? basicEvaluation.TN / negatives : 1.0;
            return (tprate + fprate) / 2.0;
        }

        public static double ComputeMultiClassAUC(int[,] confusionMatrix)
        {
            var eval = new BasicEvaluation();
            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                eval.TP += confusionMatrix[i, i];
                for (int j = 0; j < confusionMatrix.GetLength(0); j++)
                    if (i != j)
                    {
                        eval.FN += confusionMatrix[i, j];
                        eval.FP += confusionMatrix[j, i];

                        eval.TN += confusionMatrix[j, j];
                    }

            }
            return ComputeTwoClassAUC(eval);
        }
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // 1. Import or create training data

            // IDataView trainingData = Util.Load_data();
            string csv_path = @"D:\Actividades\Maestría\Segundo semestre\Machine learning\Assignment_2\VIC-master\VIC-master\VIC\data.csv";
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ModelInput>(csv_path, separatorChar: ',', hasHeader: true);

            // 2. Specify data preparation and model training pipeline
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "fingerprintEncoded", inputColumnName: "fingerprint").Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "typeEncoded", inputColumnName: "type")).Append(mlContext.Transforms.Concatenate("Features", new[] {
                "nn15",
                "nn30"    ,
                "nn45"    ,
                "nn60"    ,
                "nn75"    ,
                "nn90"    ,
                "nn105"   ,
                "nn120"   ,
                "nn135"   ,
                "nn150"   ,
                "nn165"   ,
                "nn180"   ,
                "nn195"   ,
                "nn210"   ,
                "nn225"   ,
                "nn240"   ,
                "nn255"   ,
                "nn270"   ,
                "nn285"   ,
                "nn300"   ,
                "nn315"   ,
                "nn330"   ,
                "nn345"   ,
                "nn360"   ,
                "nn375"   ,
                "nn390"   ,
                "nn405"   ,
                "nn420"   ,
                "nn435"   ,
                "nn450"   ,
                "nn465"   ,
                "nn480"   ,
                "nn495"   ,
                "nn510"   ,
                "nn525"   ,
                "nn540"   ,
                "nn555"   ,
                "nn570"   ,
                "nn585"   ,
                "nn600"   ,
                "nn607"   ,
                "nn15r"   ,
                "nn30r"   ,
                "nn45r"   ,
                "nn60r"   ,
                "nn75r"   ,
                "nn90r"   ,
                "nn105r"  ,
                "nn120r"  ,
                "nn135r"  ,
                "nn150r"  ,
                "nn165r"  ,
                "nn180r"  ,
                "nn195r"  ,
                "nn210r"  ,
                "nn225r"  ,
                "nn240r"  ,
                "nn255r"  ,
                "nn270r"  ,
                "nn285r"  ,
                "nn300r"  ,
                "nn315r"  ,
                "nn330r"  ,
                "nn345r"  ,
                "nn360r"  ,
                "nn375r"  ,
                "nn390r"  ,
                "nn405r"  ,
                "nn420r"  ,
                "nn435r"  ,
                "nn450r"  ,
                "nn465r"  ,
                "nn480r"  ,
                "nn495r"  ,
                "nn510r"  ,
                "nn525r"  ,
                "nn540r"  ,
                "nn555r"  ,
                "nn570r"  ,
                "nn585r"  ,
                "nn600r"  ,
                "nn607r"  ,
                "nn30-nn15"   ,
                "nn45-nn30"   ,
                "nn60-nn45"   ,
                "nn75-nn60"   ,
                "nn90-nn75"   ,
                "nn105-nn90"  ,
                "nn120-nn105" ,
                "nn135-nn120" ,
                "nn150-nn135" ,
                "nn165-nn150" ,
                "nn180-nn165" ,
                "nn195-nn180" ,
                "nn210-nn195" ,
                "nn225-nn210" ,
                "nn240-nn225" ,
                "nn255-nn240" ,
                "nn270-nn255" ,
                "nn285-nn270" ,
                "nn300-nn285" ,
                "nn315-nn300" ,
                "nn330-nn315" ,
                "nn345-nn330" ,
                "nn360-nn345" ,
                "nn375-nn360" ,
                "nn390-nn375" ,
                "nn405-nn390" ,
                "nn420-nn405" ,
                "nn435-nn420" ,
                "nn450-nn435" ,
                "nn465-nn450" ,
                "nn480-nn465" ,
                "nn495-nn480" ,
                "nn510-nn495" ,
                "nn525-nn510" ,
                "nn540-nn525" ,
                "nn555-nn540" ,
                "nn570-nn555" ,
                "nn585-nn570" ,
                "nn600-nn585" ,
                "nn607-nn600" ,
                "nn30r-nn15r" ,
                "nn45r-nn30r" ,
                "nn60r-nn45r" ,
                "nn75r-nn60r" ,
                "nn90r-nn75r" ,
                "nn105r-nn90r"    ,
                "nn120r-nn105r"   ,
                "nn135r-nn120r"   ,
                "nn150r-nn135r"   ,
                "nn165r-nn150r"   ,
                "nn180r-nn165r"   ,
                "nn195r-nn180r"   ,
                "nn210r-nn195r"   ,
                "nn225r-nn210r"   ,
                "nn240r-nn225r"   ,
                "nn255r-nn240r"   ,
                "nn270r-nn255r"   ,
                "nn285r-nn270r"   ,
                "nn300r-nn285r"   ,
                "nn315r-nn300r"   ,
                "nn330r-nn315r"   ,
                "nn345r-nn330r"   ,
                "nn360r-nn345r"   ,
                "nn375r-nn360r"   ,
                "nn390r-nn375r"   ,
                "nn405r-nn390r"   ,
                "nn420r-nn405r"   ,
                "nn435r-nn420r"   ,
                "nn450r-nn435r"   ,
                "nn465r-nn450r"   ,
                "nn480r-nn465r"   ,
                "nn495r-nn480r"   ,
                "nn510r-nn495r"   ,
                "nn525r-nn510r"   ,
                "nn540r-nn525r"   ,
                "nn555r-nn540r"   ,
                "nn570r-nn555r"   ,
                "nn585r-nn570r"   ,
                "nn600r-nn585r"   ,
                "nn607r-nn600r"   ,
                "d1"  ,
                "d2"  ,
                "d3"  ,
                "d4"  ,
                "d5"  ,
                "d6"  ,
                "d7"  ,
                "d8"  ,
                "d9"  ,
                "d10" ,
                "d11" ,
                "d12" ,
                "df"  ,
                "d1r" ,
                "d2r" ,
                "d3r" ,
                "d4r" ,
                "d5r" ,
                "d6r" ,
                "d7r" ,
                "d8r" ,
                "d9r" ,
                "d10r"    ,
                "d11r"    ,
                "d12r"    ,
                "dfr" ,
                "d2-d1"   ,
                "d3-d2"   ,
                "d4-d3"   ,
                "d5-d4"   ,
                "d6-d5"   ,
                "d7-d6"   ,
                "d8-d7"   ,
                "d9-d8"   ,
                "d10-d9"  ,
                "d11-d10" ,
                "d12-d11" ,
                "df-d12"  ,
                "d2r-d1r" ,
                "d3r-d2r" ,
                "d4r-d3r" ,
                "d5r-d4r" ,
                "d6r-d5r" ,
                "d7r-d6r" ,
                "d8r-d7r" ,
                "d9r-d8r" ,
                "d10r-d9r"    ,
                "d11r-d10r"   ,
                "d12r-d11r"   ,
                "dfr-d12r"    ,
                "alpha1"  ,
                "alpha2"  ,
                "alpha3"  ,
                "alpha4"  ,
                "alpha5"  ,
                "alpha6"  ,
                "alpha7"  ,
                "alpha8"  ,
                "alpha9"  ,
                "alpha10" ,
                "alpha11" ,
                "alpha12" ,
                "alphaf"  ,
                "alphan1" ,
                "alphan2" ,
                "alphan3" ,
                "alphan4" ,
                "alphan5" ,
                "alphan6" ,
                "alphan7" ,
                "alphan8" ,
                "alphan9" ,
                "alphan10"    ,
                "alphan11"    ,
                "alphan12"    ,
                "alphanf" ,
                "beta1"   ,
                "beta2"   ,
                "beta3"   ,
                "beta4"   ,
                "beta5"   ,
                "beta6"   ,
                "beta7"   ,
                "beta8"   ,
                "beta9"   ,
                "beta10"  ,
                "beta11"  ,
                "beta12"  ,
                "betaf"   ,
                "alpha1-beta1"    ,
                "alpha2-beta2"    ,
                "alpha3-beta3"    ,
                "alpha4-beta4"    ,
                "alpha5-beta5"    ,
                "alpha6-beta6"    ,
                "alpha7-beta7"    ,
                "alpha8-beta8"    ,
                "alpha9-beta9"    ,
                "alpha10-beta10"  ,
                "alpha11-beta11"  ,
                "alpha12-beta12"  ,
                "alphaf-betaf"
                })
                ).Append(mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Label"));

            /*
            var model = pipeline.Fit(trainingData);

             var scores = mlContext.BinaryClassification.CrossValidate(trainingData, pipeline, numberOfFolds: 10);
             var mean = scores.Average(x => x.Metrics.AreaUnderRocCurve);
             Console.WriteLine("El valor de AUC en la cosa esta con cross-val es: ");
             Console.WriteLine(mean);
            */


            string csv_path2 = @"D:\Actividades\Maestría\Segundo semestre\Machine learning\Assignment_2\VIC-master\VIC-master\VIC\data2.csv";
            IDataView trainingData2 = mlContext.Data.LoadFromTextFile<ModelInput2>(csv_path2, separatorChar: ',', hasHeader: true);

            // 2. Specify data preparation and model training pipeline
            var pipeline2 = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "fingerprintEncoded", inputColumnName: "fingerprint").Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "typeEncoded", inputColumnName: "type")).Append(mlContext.Transforms.Concatenate("Features", new[] {
                "nn15",
                "nn30"    ,
                "nn45"    ,
                "nn60"    ,
                "nn75"    ,
                "nn90"    ,
                "nn105"   ,
                "nn120"   ,
                "nn135"   ,
                "nn150"   ,
                "nn165"   ,
                "nn180"   ,
                "nn195"   ,
                "nn210"   ,
                "nn225"   ,
                "nn240"   ,
                "nn255"   ,
                "nn270"   ,
                "nn285"   ,
                "nn300"   ,
                "nn315"   ,
                "nn330"   ,
                "nn345"   ,
                "nn360"   ,
                "nn375"   ,
                "nn390"   ,
                "nn405"   ,
                "nn420"   ,
                "nn435"   ,
                "nn450"   ,
                "nn465"   ,
                "nn480"   ,
                "nn495"   ,
                "nn510"   ,
                "nn525"   ,
                "nn540"   ,
                "nn555"   ,
                "nn570"   ,
                "nn585"   ,
                "nn600"   ,
                "nn607"   ,
                "nn15r"   ,
                "nn30r"   ,
                "nn45r"   ,
                "nn60r"   ,
                "nn75r"   ,
                "nn90r"   ,
                "nn105r"  ,
                "nn120r"  ,
                "nn135r"  ,
                "nn150r"  ,
                "nn165r"  ,
                "nn180r"  ,
                "nn195r"  ,
                "nn210r"  ,
                "nn225r"  ,
                "nn240r"  ,
                "nn255r"  ,
                "nn270r"  ,
                "nn285r"  ,
                "nn300r"  ,
                "nn315r"  ,
                "nn330r"  ,
                "nn345r"  ,
                "nn360r"  ,
                "nn375r"  ,
                "nn390r"  ,
                "nn405r"  ,
                "nn420r"  ,
                "nn435r"  ,
                "nn450r"  ,
                "nn465r"  ,
                "nn480r"  ,
                "nn495r"  ,
                "nn510r"  ,
                "nn525r"  ,
                "nn540r"  ,
                "nn555r"  ,
                "nn570r"  ,
                "nn585r"  ,
                "nn600r"  ,
                "nn607r"  ,
                "nn30-nn15"   ,
                "nn45-nn30"   ,
                "nn60-nn45"   ,
                "nn75-nn60"   ,
                "nn90-nn75"   ,
                "nn105-nn90"  ,
                "nn120-nn105" ,
                "nn135-nn120" ,
                "nn150-nn135" ,
                "nn165-nn150" ,
                "nn180-nn165" ,
                "nn195-nn180" ,
                "nn210-nn195" ,
                "nn225-nn210" ,
                "nn240-nn225" ,
                "nn255-nn240" ,
                "nn270-nn255" ,
                "nn285-nn270" ,
                "nn300-nn285" ,
                "nn315-nn300" ,
                "nn330-nn315" ,
                "nn345-nn330" ,
                "nn360-nn345" ,
                "nn375-nn360" ,
                "nn390-nn375" ,
                "nn405-nn390" ,
                "nn420-nn405" ,
                "nn435-nn420" ,
                "nn450-nn435" ,
                "nn465-nn450" ,
                "nn480-nn465" ,
                "nn495-nn480" ,
                "nn510-nn495" ,
                "nn525-nn510" ,
                "nn540-nn525" ,
                "nn555-nn540" ,
                "nn570-nn555" ,
                "nn585-nn570" ,
                "nn600-nn585" ,
                "nn607-nn600" ,
                "nn30r-nn15r" ,
                "nn45r-nn30r" ,
                "nn60r-nn45r" ,
                "nn75r-nn60r" ,
                "nn90r-nn75r" ,
                "nn105r-nn90r"    ,
                "nn120r-nn105r"   ,
                "nn135r-nn120r"   ,
                "nn150r-nn135r"   ,
                "nn165r-nn150r"   ,
                "nn180r-nn165r"   ,
                "nn195r-nn180r"   ,
                "nn210r-nn195r"   ,
                "nn225r-nn210r"   ,
                "nn240r-nn225r"   ,
                "nn255r-nn240r"   ,
                "nn270r-nn255r"   ,
                "nn285r-nn270r"   ,
                "nn300r-nn285r"   ,
                "nn315r-nn300r"   ,
                "nn330r-nn315r"   ,
                "nn345r-nn330r"   ,
                "nn360r-nn345r"   ,
                "nn375r-nn360r"   ,
                "nn390r-nn375r"   ,
                "nn405r-nn390r"   ,
                "nn420r-nn405r"   ,
                "nn435r-nn420r"   ,
                "nn450r-nn435r"   ,
                "nn465r-nn450r"   ,
                "nn480r-nn465r"   ,
                "nn495r-nn480r"   ,
                "nn510r-nn495r"   ,
                "nn525r-nn510r"   ,
                "nn540r-nn525r"   ,
                "nn555r-nn540r"   ,
                "nn570r-nn555r"   ,
                "nn585r-nn570r"   ,
                "nn600r-nn585r"   ,
                "nn607r-nn600r"   ,
                "d1"  ,
                "d2"  ,
                "d3"  ,
                "d4"  ,
                "d5"  ,
                "d6"  ,
                "d7"  ,
                "d8"  ,
                "d9"  ,
                "d10" ,
                "d11" ,
                "d12" ,
                "df"  ,
                "d1r" ,
                "d2r" ,
                "d3r" ,
                "d4r" ,
                "d5r" ,
                "d6r" ,
                "d7r" ,
                "d8r" ,
                "d9r" ,
                "d10r"    ,
                "d11r"    ,
                "d12r"    ,
                "dfr" ,
                "d2-d1"   ,
                "d3-d2"   ,
                "d4-d3"   ,
                "d5-d4"   ,
                "d6-d5"   ,
                "d7-d6"   ,
                "d8-d7"   ,
                "d9-d8"   ,
                "d10-d9"  ,
                "d11-d10" ,
                "d12-d11" ,
                "df-d12"  ,
                "d2r-d1r" ,
                "d3r-d2r" ,
                "d4r-d3r" ,
                "d5r-d4r" ,
                "d6r-d5r" ,
                "d7r-d6r" ,
                "d8r-d7r" ,
                "d9r-d8r" ,
                "d10r-d9r"    ,
                "d11r-d10r"   ,
                "d12r-d11r"   ,
                "dfr-d12r"    ,
                "alpha1"  ,
                "alpha2"  ,
                "alpha3"  ,
                "alpha4"  ,
                "alpha5"  ,
                "alpha6"  ,
                "alpha7"  ,
                "alpha8"  ,
                "alpha9"  ,
                "alpha10" ,
                "alpha11" ,
                "alpha12" ,
                "alphaf"  ,
                "alphan1" ,
                "alphan2" ,
                "alphan3" ,
                "alphan4" ,
                "alphan5" ,
                "alphan6" ,
                "alphan7" ,
                "alphan8" ,
                "alphan9" ,
                "alphan10"    ,
                "alphan11"    ,
                "alphan12"    ,
                "alphanf" ,
                "beta1"   ,
                "beta2"   ,
                "beta3"   ,
                "beta4"   ,
                "beta5"   ,
                "beta6"   ,
                "beta7"   ,
                "beta8"   ,
                "beta9"   ,
                "beta10"  ,
                "beta11"  ,
                "beta12"  ,
                "betaf"   ,
                "alpha1-beta1"    ,
                "alpha2-beta2"    ,
                "alpha3-beta3"    ,
                "alpha4-beta4"    ,
                "alpha5-beta5"    ,
                "alpha6-beta6"    ,
                "alpha7-beta7"    ,
                "alpha8-beta8"    ,
                "alpha9-beta9"    ,
                "alpha10-beta10"  ,
                "alpha11-beta11"  ,
                "alpha12-beta12"  ,
                "alphaf-betaf"
                })
                ).Append(mlContext.Transforms.Conversion.MapValueToKey("Label")).Append(mlContext.MulticlassClassification.Trainers
                .NaiveBayes());
                //.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree()));
                //.OneVersusAll(mlContext.BinaryClassification.Trainers.FastForest()));
                //.OneVersusAll(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression())); 
                //.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron()));
                //.OneVersusAll(mlContext.BinaryClassification.Trainers.LinearSvm()));
                //.LbfgsMaximumEntropy());


            var scores2 = mlContext.MulticlassClassification.CrossValidate(trainingData2, pipeline2, numberOfFolds: 10);
            var predictions2 = scores2[0];
            var mean2 = scores2.Max(x => x.Metrics.MacroAccuracy);
            Console.WriteLine("El valor de AUC en la cosa esta con cross-val es: ");
            Console.WriteLine(mean2);
            Console.WriteLine(predictions2.Metrics.ConfusionMatrix.GetFormattedConfusionTable());
            int[,] matriz = new int[3, 3];
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0;j < 3; j++)
                {
                    matriz[i, j] = (int)predictions2.Metrics.ConfusionMatrix.GetCountForClassPair(i, j);
                }
            }
            Console.WriteLine(ComputeMultiClassAUC(matriz));
            /*
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Console.Write(matriz[i, j]+" ");
                }
                Console.WriteLine();
            }
            */



            // 3. Train model
            //var model = pipeline.Fit(trainingData);

        }




    }


}
