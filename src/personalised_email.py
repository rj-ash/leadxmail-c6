from dotenv import load_dotenv
import os
import json
import warnings
import io
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
import time
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

def query_llm_api(prompt: str) -> str:
    """
    Get response from OpenAI API using the provided prompt and temperature
    
    Args:
        prompt (str): The prompt to send to the model
        
    Returns:
        str: The model's response or a mock response if API is unavailable
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
            
        model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0.7,
            max_tokens=4096,
        )
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=30),
            retry=retry_if_exception_type((RateLimitError, APIError)),
            after=lambda retry_state: time.sleep(2)
        )
        def _invoke_model():
            try:
                result = model.invoke(prompt)
                return result.content
            except (RateLimitError, APIError) as e:
                print(f"Rate limit hit, retrying after delay... Error: {str(e)}")
                raise
            except Exception as e:
                print(f"Error invoking model: {str(e)}")
                raise
                
        return _invoke_model()
    except Exception as e:
        print(f"API Error: {str(e)}")
        # Return a mock response when API is unavailable
        return json.dumps({
            "subject": "Mock Email Subject",
            "body": f"""Dear [Name],

I was just walking through your LinkedIn profile and noticed your impressive background in [Current experience] at [Company]. 

Our platform offers real-time data processing and visualization capabilities that could significantly enhance your current operations. With features like advanced machine learning algorithms and customizable dashboards, we've helped companies achieve 40% faster decision-making and 60% reduction in manual data analysis.

I'd love to schedule a brief demo to show you how our solution could benefit [Company]. Would you be available for a 30-minute call next week?

Best regards,
[Your Name]
[Your Position]
[Your Company]"""
        })


# Define the email generation prompt
prompt = """You are an B2B expert marketer. Based on the list of leads, their details and the product document provides , write a personalized email to the lead (whose details are provided) in a concise manner, keeping in mind they have limited time to read the email. Each email should have:  1. **Subject:** The subject line should be catchy, highly personalized and relevant to the lead's profile. 2. **Body:** The email body should be structured naturally, incorporating:  - A personalized introduction  - Key product features, unique selling points (USPs), and benefits tailored to the lead's profile, without explicitly labeling them as separate sections.  - A persuasive closing with a compelling call to action (CTA) to encourage engagement (e.g., scheduling a demo, signing up for a trial, etc.).  Ensure that all elements (introduction, product benefits, CTA) are woven into the body fluidly without explicit section headers like 'Introduction' or 'Call to Action.'  Also, just provide the email content as requested. Do not include any introductory or concluding remarks like 'Here's your template' or 'Let me know if you need anything else.' The response should contain **only the emails** in the specified format. At the end of mail, Greet with regards, your name and contact details (take from product pdf, if provided, else leave as template). Make sure generated emails for different leads stay diverse, not share common template and are highly personalised and relevant. Also do not link the product with the lead in the fields that are not fitting to the product, try match features of product to the fitted area of lead other wise need not to mention. You can trade off formality for creativity as long as you are not sounding too informal and unprofessional to grab the attention of lead."""
personalisation_convention = """ Take a look at the details of lead primarily on their experience, education, current company's industry and overview then skills and bio and then match it with the product features and benefits. In first paragraph greet them and talked about their details, avoid common templates (like I hope this email finds you well) and write a catcy introduction but natural (look like written by a human), try different variations. Include like I was going through your linkedin profile and noticed this this and this (this should be a common interest of the lead and your product). In next paragraph, talk about potential problems existing in his industry (Refer prvided product document and take the problem from that document) and then introduce the product and clearly articulate how the product addresses a specific problem or enhances their current processes, supported by concise benefits. Also mention about the past achievements and works of product if mentioned in document. At last conclude this with a CTA, a specific time for a demo or provide a scheduling link, making it easy for the recipient to respond. Express appreciation for their time and consideration, and indicate your anticipation of their response. At last put your signature: Include full name (Template), position, company name(Template), contact information (Template), and a professional sign-off. Refrain from fitting the product features to the lead details, if it is not fitting, do not mention it."""

def generate_email_for_single_lead(lead_details: dict, product_details: str) -> dict:
    """
    Generate a personalized email for a single lead
    
    Args:
        lead_details (dict): Dictionary containing lead details with keys:
            - Name: str
            - Lead_ID: int
            - Current experience: str
            - Education: str
            - Company: str
            - Company overview: str
            - Company industry: str
        product_details (str): Product documentation/information
        
    Returns:
        dict: Dictionary containing 'subject', 'body', and 'lead_id' of the email
    """
    # Construct the prompt with lead details and product information
    instructed_prompt = f"""
Based on the following lead details, product information and the prompt, generate a personalized email and follow the personalisation convention for the lead.

Lead Details: {lead_details}
Product Information: {product_details}

prompt: ```{prompt}```
personalisation convention: ```{personalisation_convention}```

Generate a personalized email following the format specified in the prompt above.
Return the results in a dictionary with three keys:
1. 'subject': The email subject line
2. 'body': The email body content

Make sure the email is unique and highly personalized based on the lead's profile and relevant product features.
"""
    
    # Get the response from OpenAI
    email_response = query_llm_api(instructed_prompt)
    
    # Parse the response into a dictionary
    try:
        # Clean the response and safely parse it
        email_response = email_response.strip().lower().replace('json', '', 1).strip().strip('`').strip()
        result_dict = json.loads(email_response)
        
        # Ensure the response has the correct structure
        if not all(key in result_dict for key in ['subject', 'body']):
            raise ValueError("Response missing required keys: 'subject' and 'body'")
            
        # Add lead_id to the result
        result_dict['lead_id'] = lead_details.get('Lead_ID')
            
    except json.JSONDecodeError:
        try:
            result_dict = eval(email_response)
            if not all(key in result_dict for key in ['subject', 'body']):
                raise ValueError("Response missing required keys: 'subject' and 'body'")
            # Add lead_id to the result
            result_dict['lead_id'] = lead_details.get('Lead_ID')
        except Exception as e:
            raise ValueError(f"Failed to parse the OpenAI response: {email_response}") from e
    
    return result_dict

def generate_email_for_multiple_leads(leads_list: list, product_details: str) -> list:
    """
    Generate personalized emails for multiple leads
    
    Args:
        leads_list (list): List of dictionaries, where each dictionary contains lead details with keys:
            - Name: str
            - Lead_ID: int
            - Current experience: str
            - Education: str
            - Company: str
            - Company overview: str
            - Company industry: str
        product_details (str): Product documentation/information
        
    Returns:
        list: List of dictionaries, each containing 'subject', 'body', and 'lead_id' of the email
    """
    if not leads_list:
        raise ValueError("No leads provided in the list")
    
    all_emails = []
    
    # Process in batches of 5 leads
    batch_size = 5
    for i in range(0, len(leads_list), batch_size):
        batch_leads = leads_list[i:i + batch_size]
        
        # Process each lead in the batch
        batch_emails = []
        for lead in batch_leads:
            try:
                # Generate email for this lead
                result = generate_email_for_single_lead(lead, product_details)
                batch_emails.append(result)
            except Exception as e:
                print(f"Error processing lead {lead.get('Name', 'Unknown')}: {str(e)}")
                batch_emails.append({
                    'subject': 'Error generating email',
                    'body': f'Error generating personalized email: {str(e)}',
                    'lead_id': lead.get('Lead_ID')
                })
        
        # Add the batch results to our main list
        all_emails.extend(batch_emails)
        
        # Add delay between batches to avoid rate limits
        if i + batch_size < len(leads_list):
            time.sleep(2)
    
    return all_emails

def main():
    """
    Main function to test personalized email generation with sample data
    """
    # Sample product details
    product_details = """
    Product: AI-Powered Analytics Platform
    
    Key Features:
    - Real-time data processing and visualization
    - Advanced machine learning algorithms
    - Customizable dashboards
    - Enterprise-grade security
    - 24/7 technical support
    
    Benefits:
    - 40% faster decision-making
    - 60% reduction in manual data analysis
    - Seamless integration with existing systems
    - Scalable architecture
    - Competitive pricing
    """

    # Sample lead details
    sample_lead = {
        'Name': 'John Doe',
        'Lead_ID': 1010101,
        'Current experience': 'Senior Data Scientist with 5 years of experience in machine learning and big data analytics',
        'Education': 'M.S. in Computer Science from Stanford University',
        'Company': 'TechCorp Inc.',
        'Company overview': 'Leading provider of enterprise software solutions',
        'Company industry': 'Technology and Software Development'
    }

    # Sample list of leads
    sample_leads = [
        {
            'Name': 'John Doe',
            'Lead_ID': 7847638,
            'Current experience': 'Senior Data Scientist with 5 years of experience in machine learning and big data analytics',
            'Education': 'M.S. in Computer Science from Stanford University',
            'Company': 'TechCorp Inc.',
            'Company overview': 'Leading provider of enterprise software solutions',
            'Company industry': 'Technology and Software Development'
        },
        {
            'Name': 'Jane Smith',
            'Lead_ID': 28748378,
            'Current experience': 'Product Manager with 3 years of experience in enterprise software',
            'Education': 'MBA from Harvard Business School',
            'Company': 'Enterprise Solutions Ltd.',
            'Company overview': 'Provider of enterprise resource planning software',
            'Company industry': 'Enterprise Software'
        }
    ]

    # print(generate_email_for_single_lead(sample_lead, product_details))
    print(generate_email_for_multiple_leads(sample_leads, product_details))


if __name__ == "__main__":
    main()


