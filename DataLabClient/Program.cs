using DataLabServer;
using Grpc.Net.Client;
using System;
using System.Net.Http;

namespace DataLabClient
{
    class Program
    {
        static void Main(string[] args)
        {
            var httpClientHandler = new HttpClientHandler();
            httpClientHandler.ServerCertificateCustomValidationCallback = HttpClientHandler.DangerousAcceptAnyServerCertificateValidator;
            var httpClient = new HttpClient(httpClientHandler);
            var options = new GrpcChannelOptions() { HttpClient = httpClient };

            using var channel = GrpcChannel.ForAddress("https://10.0.0.21:5001", options);
            var client = new WorkerEndpoints.WorkerEndpointsClient(channel);

            var wi = new WorkerInfo();
            wi.Name = "bananas";

            var reply = client.RegisterWorker(wi);

            Console.WriteLine("reply id {0}", reply.Id);
            Console.ReadLine();
        }
    }
}
