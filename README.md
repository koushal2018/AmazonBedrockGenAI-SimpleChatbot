# AmazonBedrockGenAI-SimpleChatbot
 AmazonBedrockGenAI chatbot uses RAG architecture, with Faiss for response retrieval from indexed convos &amp; LangChain to refine replies, alongside Bedrock API access to Megatron Turing for high quality generation. Built with Streamlit for web interaction.

I assume that you already have set-up with Bedrock IAM permissions and have access to the bedrock models. 
If not, please refer to AWS documentation the IAM permissions and access to the Amazon Bedrock Models. 

Git Clone


```
gh repo clone koushal2018/AmazonBedrockGenAI-SimpleChatbot
```
or 
```
https://github.com/koushal2018/AmazonBedrockGenAI-SimpleChatbot.git
```

unzip the folder 

```
cd AmazonBedrockGenAI-SimpleChatbot
```


Dependencies

Make sure you have the latest version of Boto3:

```
pip install -r requirements.txt
```

Open the file setup.ipynb in your IDE and run the code, you can choose to run Run All or run section by section 

Once you have completed running

From the terminal - type this to run your application 

```
streamlit run app.py