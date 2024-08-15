import openai
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Check if a fine-tuned model already exists
fine_tuned_model_id = None
fine_tuning_jobs = openai.FineTuningJob.list()

for job in fine_tuning_jobs['data']:
    if job['status'] == 'succeeded' and job.get('fine_tuned_model'):
        fine_tuned_model_id = job['fine_tuned_model']
        print(f"Using existing fine-tuned model: {fine_tuned_model_id}")
        break

if not fine_tuned_model_id:
    print("No fine-tuned model found, creating a new one.")
    
    # Upload the JSONL file
    response = openai.File.create(file=open("../fine_tuneData/Dataset.jsonl", "rb"), purpose="fine-tune")
    
    file_id = response["id"]
    print("File ID:", file_id)

    # Create a fine-tuning job
    fine_tune_response = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
    fine_tune_job_id = fine_tune_response["id"]
    print(f"Fine-Tuning Job ID: {fine_tune_job_id}")

    # Wait for fine-tuning to complete
    while True:
        job_status = openai.FineTuningJob.retrieve(fine_tune_job_id)
        print("Job Status:", job_status)
        if job_status['status'] == 'succeeded':
            fine_tuned_model_id = job_status['fine_tuned_model']
            print(f"Fine-tuned model created: {fine_tuned_model_id}")
            break
        elif job_status['status'] in ['failed', 'cancelled']:
            raise ValueError(f"Fine-tuning job failed or was cancelled: {job_status}")
        time.sleep(30)  # Wait before checking again

# Save the fine-tuned model ID to a file
with open("fine_tuned_model_id.txt", "w") as f:
    f.write(fine_tuned_model_id)
    print(f"Fine-tuned model ID saved to fine_tuned_model_id.txt")
