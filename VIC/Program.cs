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
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // 1. Import or create training data

           // IDataView trainingData = Util.Load_data();
            string csv_path = @"D:\Actividades\Maestría\Segundo semestre\Machine learning\Assignment_2\VIC-master\VIC-master\VIC\data.csv";
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ModelInput>(csv_path, separatorChar: ',', hasHeader: true);
    
            // 2. Specify data preparation and model training pipeline
            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "fingerprintEncoded", inputColumnName: "fingerprint").Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "minutiaEncoded", inputColumnName: "minutia")).Append(mlContext.Transforms.Concatenate("Features", new[] { 
                "nn15",
                "nn255"       
                })
                ).Append(mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Label"));

            var model = pipeline.Fit(trainingData);

            var scores = mlContext.BinaryClassification.CrossValidate(trainingData, pipeline, numberOfFolds: 10);
            var mean = scores.Average(x => x.Metrics.AreaUnderRocCurve);
            Console.WriteLine("El valor de AUC en la cosa esta con cross-val es: ");
            Console.WriteLine(mean);

          

            string csv_path2 = @"D:\Actividades\Maestría\Segundo semestre\Machine learning\Assignment_2\VIC-master\VIC-master\VIC\data2.csv";
            IDataView trainingData2 = mlContext.Data.LoadFromTextFile<ModelInput2>(csv_path2, separatorChar: ',', hasHeader: true);

            // 2. Specify data preparation and model training pipeline
            var pipeline2 = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "fingerprintEncoded", inputColumnName: "fingerprint").Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "minutiaEncoded", inputColumnName: "minutia")).Append(mlContext.Transforms.Concatenate("Features", new[] {
                "nn15",
                "nn255"
                })
                ).Append(mlContext.Transforms.Conversion.MapValueToKey("Label")).Append(mlContext.MulticlassClassification.Trainers
                .OneVersusAll(mlContext.BinaryClassification.Trainers.FastForest()));


            //var model2 = pipeline2.Fit(trainingData2);

            var scores2 = mlContext.MulticlassClassification.CrossValidate(trainingData2, pipeline2, numberOfFolds: 10);
            var predictions2 = scores2.Count;
            var mean2 = scores2.Average(x => x.Metrics.MacroAccuracy);
            Console.WriteLine("El valor de AUC en la cosa esta con cross-val es: ");
            Console.WriteLine(mean2);
            Console.WriteLine(predictions2);




            // 3. Train model
            //var model = pipeline.Fit(trainingData);

        }




    }


}
