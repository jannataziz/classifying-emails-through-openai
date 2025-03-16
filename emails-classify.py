# Import required libraries
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Load the email dataset
emails_df = pd.read_csv('email_categories_data.csv')
# Display the first few rows of our dataset
print("Preview of our email dataset:")
print(emails_df)

load_dotenv()

openai.api_key= os.environ["OPENAI_API_KEY"]

prompt= """ You are an email classifier.Your task is to classify each email into exactly one category:Priority,Updates, or Promotions. 
Only response with one word: Priority, Updates, or Promotions.  

Rules:
- Priority: Urgent matters, meetings, deadlines, and important team communications.
- Updates: General information, newsletters, product launches, non-urgent announcements
- Promotions: Sales ,marketing offers , advertisements,deals

Below are some examples to help you understand how to classify emails:
Example 1:
Email: Server Maintenance Required.
Classification: Priority

Example 2:
Email: New Product Launch Invitation
Classification: Updates

Example 3:
Email: Flash Sale - 24 Hours Only!
Classification: Promotions

Classify this email into either Priority, Pomotions or Updates:
Email: {message}
Classification: """

def process_email(message,prompt):
    input_prompt=f"{prompt}{message}"
    client=OpenAI(api_key=openai.api_key)
    response= client.chat.completions.create(
        model='gpt-4-turbo',
        messages=[{'role':'system', 'content':'You are an AI email classifier.'},
                  {'role':'user','content':input_prompt}],
    )
    data = response.choices[0].message.content
    print(data)
    model_output=data.split(":")[0].strip()
    return model_output
results=[]
test_emails=emails_df[3:7]
for index,row in test_emails.iterrows():
    email_content=row['email_content']
    expected_category=row['expected_category']
    output=process_email(email_content,prompt)
    results.append({
        'email_content': email_content,
        'expected_category': expected_category,
        'model_output': output
    })

    results_df = pd.DataFrame(results)
print(results_df)
