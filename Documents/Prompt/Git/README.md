

📚 ***Bookino***
Book Recommendation Chatbot

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

***Contact Info***

gitHub:https://github.com/ShimaBolboli
LinkedIn:www.linkedin.com/in/shima-bolboli![image](https://github.com/user-attachments/assets/bf668963-782b-4410-87b1-bdfbe0c679c1)
Email:s.bolboli@gmail.com

--------------------------------------------------
lic
***YouTube Link***

https://youtu.be/XfZ2e17V3dQ


