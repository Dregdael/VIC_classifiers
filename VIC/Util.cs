using System.IO;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace VIC
{
    class Util
    {
        public static void Arff2csv(string arff_path = @"D:\Actividades\Maestría\Segundo semestre\Machine learning\Assignment_2\VIC-master\VIC-master\VIC\data.arff")
        {
            string[] text = System.IO.File.ReadAllText(arff_path).Split("@data");
            Regex rx = new Regex(@"@attribute (?<attr_name>[\w-]+|('[\w-]+( [\w-]+)+')) .+", RegexOptions.Compiled);
            string header = "";
            foreach (Match m in rx.Matches(text[0]))
            {
                header += ',' + m.Groups["attr_name"].Value;
            }
            text[1].Remove('?');
            string output_path = Path.Join(Path.GetDirectoryName(arff_path), Path.GetFileNameWithoutExtension(arff_path) + ".csv");
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(output_path))
            {
                file.WriteLine(header.Substring(1));
                file.Write(text[1]);
            }
        }

        public static IDataView Load_data(string csv_path = @"D:\Actividades\Maestría\Segundo semestre\Machine learning\Assignment_2\VIC-master\VIC-master\VIC\data.csv")
        {
            var mlContext = new MLContext();
            IDataView data = mlContext.Data.LoadFromTextFile<MinutiaData>(csv_path, separatorChar: ',', hasHeader: true);
            return data;
        }

        /*
        public static IDataView get2Clusters(IDataView data, string column_name="score")
        {
            data.Sort = column_name;
            return data;
        }
        */
    }
}
