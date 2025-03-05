# Run the following cells first
# Install necessary packages

# Import required libraries
import pandas as pd
from llama_cpp import Llama

# Load the email dataset
emails_df = pd.read_csv('email_categories_data.csv')
# Display the first few rows of our dataset
print("Preview of our email dataset:")
print(emails_df)

prompt= """ You need to classify emails into Priority,Updates and Permissions like below examples given:
Example 1:
Email: Server Maintenance Required.
Response: Priority

Example 2:
Email: New Product Launch Invitation
Response: Updates

Example 3:
Email: Flash Sale - 24 Hours Only!
Response: Promotions

Now classify this email whether its priority, promotions or updates:
Email: 
 """
llm=Llama(model_path="model.gguf")

def process_email(llm,message,prompt):
    input_prompt=f"{prompt}{message}"
    response=llm.create_completion(input_prompt,max_tokens=5,temperature=0.1)
    print(response)
    model_output=response['choices'][0]['text'].split(' ')
    return model_output[1]
results=[]
test_emails=emails_df[3:7]
for index,row in test_emails.iterrows():
    email_content=row['email_content']
    expected_category=row['expected_category']
    output=process_email(llm,email_content,prompt)
    results.append({
        'email_content': email_content,
        'expected_category': expected_category,
        'model_output': output
    })

    results_df = pd.DataFrame(results)

    print(results_df)