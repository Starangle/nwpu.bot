namespace 编程之美初赛.Dialogs
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using System.Web;
    using Microsoft.Bot.Builder.Dialogs;
    using Microsoft.Bot.Builder.FormFlow;
    using Microsoft.Bot.Builder.Luis;
    using Microsoft.Bot.Builder.Luis.Models;
    using Microsoft.Bot.Connector;
    using System.Net;
    using System.Web.Script.Serialization;

    [LuisModel("3959bf61-312e-4396-b29d-c0e6029b4b2b", "a47c94ce176c485ca5898bd3920a23c4")]
    [Serializable]
    public class RootLuisDialog : LuisDialog<object>
    {
        class QnAMakerAnswer
        {
            public string answer { get; set; }
            public string score { get; set; }
        }

        private string Query(string question,bool only_answer=true)
        {
            string responseString = string.Empty;

            var query = question; //User Query
            var knowledgebaseId = "02a3356e-b886-435b-96d2-8904310be161"; // Use knowledge base id created.
            var qnamakerSubscriptionKey = "f7ad3210dcc9446d871b7f1cd609059d"; //Use subscription key assigned to you.

            //Build the URI
            Uri qnamakerUriBase = new Uri("https://westus.api.cognitive.microsoft.com/qnamaker/v1.0");
            var builder = new UriBuilder($"{qnamakerUriBase}/knowledgebases/{knowledgebaseId}/generateAnswer");

            //Add the question as part of the body
            var postBody = $"{{\"question\": \"{query}\"}}";

            //Send the POST request
            using (WebClient client = new WebClient())
            {
                //Set the encoding to UTF8
                client.Encoding = System.Text.Encoding.UTF8;

                //Add the subscription key header
                client.Headers.Add("Ocp-Apim-Subscription-Key", qnamakerSubscriptionKey);
                client.Headers.Add("Content-Type", "application/json");
                responseString = client.UploadString(builder.Uri, postBody);
            }
            if(only_answer)
            {
                JavaScriptSerializer translater = new JavaScriptSerializer();
                responseString = translater.Deserialize<QnAMakerAnswer>(responseString).answer;
            }
            return responseString;
        }

        [LuisIntent("")]
        public async Task None(IDialogContext context, LuisResult result)
        {
            string message = result.TopScoringIntent.Intent;
            message = "学校简介";
            message = Query(message);
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }

        [LuisIntent("问候")]
        public async Task 问候(IDialogContext context, LuisResult result)
        {
            string message = $"你好，有什么可以帮你的？";
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }

        [LuisIntent("Q人物")]
        public async Task Q人物(IDialogContext context, LuisResult result)
        {
            string message = $"询问人物";
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }
    }
}