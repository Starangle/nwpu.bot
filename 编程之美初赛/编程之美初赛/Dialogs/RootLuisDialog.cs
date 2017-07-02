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

        string zone, obj, tim, job, cul, que, awa, kin;

        private string Query(string question, bool only_answer = true)
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
            if (only_answer)
            {
                JavaScriptSerializer translater = new JavaScriptSerializer();
                responseString = translater.Deserialize<QnAMakerAnswer>(responseString).answer;
                if (responseString == "No good match found in the KB")
                {
                    responseString = "对不起，暂时无法回答该问题！\n";
                }
            }
            return responseString;
        }

        [LuisIntent("")]
        public async Task None(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("问题", out enitiy))
            {
                que = enitiy.Entity.ToString();
            }
            string message = Query(que);
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
        [LuisIntent("域获得过多少奖励")]
        public async Task 域获得过多少奖励(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("域", out enitiy))
            {
                zone = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("奖励", out enitiy))
            {
                awa = enitiy.Entity.ToString();
            }

            string message = zone + "获得过多少" + awa;
            message = Query(message);
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }

        [LuisIntent("什么时间成为客体")]
        public async Task 什么时间成为客体(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("时间", out enitiy))
            {
                tim = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("客体", out enitiy))
            {
                obj = enitiy.Entity.ToString();
            }

            string message = "什么时候成为" + obj;
            message = Query(message);
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }

        [LuisIntent("域有多少客体")]
        public async Task 域有多少客体(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("域", out enitiy))
            {
                zone = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("客体", out enitiy))
            {
                obj = enitiy.Entity.ToString();
            }

            string message = zone + "有多少" + obj;
            message = Query(message);
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }
        [LuisIntent("历史上培养出多少客体")]
        public async Task 历史上培养出多少客体(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("域", out enitiy))
            {
                zone = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("客体", out enitiy))
            {
                obj = enitiy.Entity.ToString();
            }

            string message = zone + "培养出多少" + obj;
            message = Query(message);
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }

        [LuisIntent("文化是什么")]
        public async Task 文化是什么(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("域", out enitiy))
            {
                zone = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("文化", out enitiy))
            {
                cul = enitiy.Entity.ToString();
            }

            string message = zone + "的" + cul;
            message = Query(message);
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }
        [LuisIntent("时间.域的职务是谁")]
        public async Task 时间_域的职务是谁(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("域", out enitiy))
            {
                zone = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("职务", out enitiy))
            {
                job = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("时间", out enitiy))
            {
                tim = enitiy.Entity.ToString();
            }
            string message = tim + zone + "的" + job + "是谁";
            message = Query(message);
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }
        [LuisIntent("查询客体集合")]
        public async Task 查询客体集合(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("客体", out enitiy))
            {
                obj = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("域", out enitiy))
            {
                zone = enitiy.Entity.ToString();
            }
            string message = zone + obj + "查询";
            message = Query(message);
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }
        [LuisIntent("客体类别_域判定")]
        public async Task 客体类别_域判定(IDialogContext context, LuisResult result)
        {
            EntityRecommendation enitiy;
            if (result.TryFindEntity("客体", out enitiy))
            {
                obj = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("域", out enitiy))
            {
                zone = enitiy.Entity.ToString();
            }
            if (result.TryFindEntity("类别", out enitiy))
            {
                kin = enitiy.Entity.ToString();
            }
            string message = zone + kin + "查询";
            message = Query(message);

            if (message.Contains(obj))
                message = "是";
            else message = "不是";
                
            await context.PostAsync(message);
            context.Wait(this.MessageReceived);
        }
    }
}