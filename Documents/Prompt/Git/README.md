
🦜🔗
🔧🤖
**Book Recommendation Chatbot**

Bookino is an AI-powered chatbot designed to provide personalized book recommendations based on user queries. This application leverages LangChain, Pinecone, and OpenAI's fine-tuned models to generate accurate and relevant responses.
---------------------------------------
***Features***
Features
Natural Language Processing (NLP): Uses advanced NLP models to understand user queries and provide appropriate book recommendations.
Fine-Tuned Model: Incorporates a custom fine-tuned model to enhance the accuracy and relevance of the recommendations.
Pinecone Integration: Utilizes Pinecone for efficient vector storage and similarity search to find relevant content from the book dataset.
Streamlit Interface: Provides an interactive and user-friendly interface using Streamlit, allowing users to ask questions and receive recommendations in real-time.


---------------------------------
***Installation***

Prerequisites

Python 3.8+
A Pinecone account with API access
OpenAI API key
A fine-tuned model (can be created using the provided code)
Streamlit for the web interface


1-Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```
2- Install the required packages:
```
pip install -r requirements.txt
````
3- Define Pinecone API Key

https://app.pinecone.io/organizations/-O-BeAvN27eaGSzY1U3n/projects/5bf92691-726e-439e-8a2a-08ff42d4e99d/keys

4- Define OpenAI API Key

https://platform.openai.com/api-keys

5-Set up Pinecone environment variables:

,,,
export PINECONE_API_KEY='your_pinecone_api_key'
export PINECONE_ENVIRONMENT='your_pinecone_environment'
,,,,

6- Prepare the fine-tuned model:
python fine_tune_model.py

The model ID will be saved in fine_tuned_model_id.txt.

7-Run the application
streamlit run openAI-json.py

8-Access the chatbot:

Open your web browser and go to http://localhost:8501 to interact with Bookino.

***NOTE:*** I have 2 model of fine-tune:
Model 1 (openAI.py): Created through OpenAI's dashboard, using the provided model ID in your code.
Model 2 (openAI-json.py): Created by uploading a dataset programmatically via fine_tune_model.py, also using the model ID in your code.

Both models use OpenAI's fine-tuning services but were set up through different methods—one via the dashboard and the other via a custom script.

Dashboard: Easier to use with less technical overhead.Best for quick, one-off fine-tuning with minimal coding.

Programmatic: Offers more control and flexibility but requires more effort.


-------------------------------------------------
***Usage***

1- Ensure your dataset (For example books.txt) is in the project directory.
you can download dataset from website like Kaggle 

2- Run the Streamlit app:

```
streamlit run fine_name.py
```

3-Open your browser and navigate to http://localhost:8501 to interact with the chatbot.
-----------------------------------------------
***How It Works***

***Loading and Splitting Documents***

The application loads a text file (book.txt) and splits it into chunks using LangChain's CharacterTextSplitter.

***Vectorization and Storage***

The text chunks are embedded using HuggingFaceEmbeddings and stored in Pinecone, enabling efficient similarity search.

***Fine-Tuned Model***

A fine-tuned model is used to generate responses based on the context retrieved from Pinecone. If no relevant context is found, the chatbot will return "No data found."

***User Interaction***

The chatbot interface is built with Streamlit, providing a simple and interactive way for users to ask questions and receive recommendations.


***YouTube Link***

https://youtu.be/XfZ2e17V3dQ
