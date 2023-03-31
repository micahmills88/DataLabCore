using System;
using System.Collections.Generic;
using System.Linq;
using DataLabCore;

namespace DataLabConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            string uri = @"mongodb://10.0.0.20:27017";
            string modelName = "TCN_RUN_" + Guid.NewGuid().ToString("N");
            var datasource = new DataSource_TEXT(64, 50000);

            string modelid = "5e971d02b50a05302c71cadd";
            var runner = ModelRunner.LoadFromDatabase(uri, modelid, ControllerType.CUDA, 64, 64, 1, 1);

            TensorController tc = new TensorController(ControllerType.CUDA);
            int sampleLength = 150;

            var rawChars = datasource.GetSeedData();
            var finalOutput = new List<char>();
            var rawdata = datasource.CharToVect(rawChars);
            Tensor msg = new Tensor(tc, 64, 64, rawdata);
            var sampleList = rawdata.ToList();

            Console.WriteLine("Seed Data:\n{0}", new string(rawChars));
            while (true)
            {
                for (int i = 0; i < sampleLength; i++)
                {
                    var output = runner.Process(msg);
                    output.SynchronizeToLocal();
                    var nextchar = datasource.GenerateCharProbability(output.Data);
                    finalOutput.Add(nextchar);
                    sampleList.RemoveAt(0);
                    sampleList.AddRange(output.Data);
                    msg.Data = sampleList.ToArray();
                    msg.SynchronizeToRemote();
                }

                
                Console.WriteLine("Sample test output:\n=============================================================");
                var outchars = finalOutput.ToArray();
                var outstring = new string(outchars);
                Console.WriteLine(outstring);

                Console.WriteLine("=============================================================\nPress Enter To Continue...");
                Console.ReadLine();

                finalOutput.Clear();
            }

                        
        }
    }
}
