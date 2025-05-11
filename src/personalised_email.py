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
[Your Company]
[Your Contact Information]
Replace [Your Name], [Your Position], [Your Company] with your name, position and company name given in the product_details
Replace [Your Contact Information] with your contact information given in the product_details"""
        })


# Define the email generation prompt
prompt = """You are an B2B expert marketer. Based on the list of leads, their details and the product document provides , write a personalized email to the lead  in a concise manner, keeping in mind they have limited time to read the email. It shoud be casual and not too formal. Avoide using unnecessary words and too much buttering. Be casual, friendly and direct. Each email should have:  1. **Subject:** The subject line should be catchy, highly personalized and relevant to the lead's profile. 2. **Body:** Start with telling the leads that you were going through your profile and talk a bit about his company. Then in next para, talk about problems in the industry of his company and introduce your product stating how it solve this problem. Get the details from product_details section. Then a compelling CTA asking him to schedule a meet or a call to discuss it further. Keep it short direct, casual, catchy and dont use too much formal jargonand avoid common templates (like I hope this email finds you well, etc). It should look as it is written by a human after carefully studying his profile and his company's details."""
style = """
Do not use any common templates like I hope this email finds you well, etc.
Do not use too much buttering and unnecessary words.
Be casual, friendly and direct. Use human like tone and language. Follow this style:
1. First Person Pronouns: I, me, my, mine, we, us, our, ours. Write as if you are the one talking to the lead and do not use generalised statements.
2. Fillers & Disfluencies
Spoken or informal written human language often includes:
	•	uh, um, like, you know, kinda, sorta, actually, basically, literally
	•	contractions: gonna, wanna, gotta, ain’t
3. Personal Experience Markers
	•	I think, I believe, I feel, in my opinion
	•	yesterday, last week, when I was in school
	•	my friend, my boss, my mom said
4. Typos and Misspellings
Humans often make minor spelling or grammatical errors:
	•	definately → definitely
	•	alot → a lot
	•	seperate → separate
5. Emotional/Spontaneous Expressions
Humans express feelings impulsively or with less filter:
	•	wow, amazing, omg, haha, lol, damn, yay
	•	love it, hate that, so cool, super weird
"""
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

prompt: ```{prompt}```
style: ```{style}```
Lead Details: {lead_details}
Product Information: {product_details}



Generate a personalized email following the format specified in the prompt above.
Return the results in a dictionary with these keys:
1. 'subject': The email subject line
2. 'body': The email body content

Make sure the email is unique and highly personalized based on the lead's profile and relevant product features.
"""
    
    # Get the response from OpenAI
    email_response = query_llm_api(instructed_prompt)
    
    # Parse the response into a dictionary
    try:
        # Clean the response and safely parse it
        email_response = email_response.strip().replace('json', '', 1).strip().strip('`').strip()
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
#     # Sample product details
    product_details = """
InvestorBase is an AI-powered platform designed to revolutionize deal flow management for venture capitalists (VCs). It automates the evaluation of pitch decks, enabling VCs to identify high-potential opportunities efficiently, reducing manual workloads, and minimizing missed prospects. The platform streamlines investment decision-making through AI-driven analysis, real-time market validation, and customizable deal scoring models.
Key Features & Capabilities
Comprehensive Pitch Deck Analysis
Executive Summary Assessment: Evaluates the clarity of value proposition and business model.
Market Size Validation: Cross-references TAM/SAM/SOM claims with industry databases.
Financial Model Scrutiny: Flags unrealistic growth projections and validates unit economics.
Competitive Landscape Analysis: Maps startup positioning against established players.
Team Background Verification: Assesses founder experience relevance to venture success.
Advanced Market Validation
Industry Growth Trends: Overlays startup trajectory against sector forecasts.
Regulatory Impact Assessment: Identifies compliance challenges in regulated industries.
Geographic Expansion Feasibility: Evaluates market entry barriers for scaling plans.
Technology Adoption Curves: Compares innovation timing with market readiness.
Customer Acquisition Cost Benchmarking: Compares CAC/LTV ratios with industry standards.
Customized Deal Prioritization
Investment Thesis Alignment: Scores opportunities against firm-specific criteria.
Portfolio Fit Analysis: Identifies synergies with existing investments.
Stage-Specific Metrics: Adapts evaluation criteria based on startup maturity.
Risk-Adjusted Return Potential: Weighs opportunities by both upside and risk factors.
Follow-on Investment Planning: Highlights portfolio companies ready for additional funding.

Use Cases
Efficiency Gains for Venture Capitalists
Reduces initial screening time by 70-80% through automated analysis.
Increases deal throughput capacity by 3-5x without adding staff.
Enables analysts to focus on high-value activities (founder interactions, due diligence) instead of basic screening.
Cuts meeting time spent discussing marginal deals by 40-50%.
Enhanced Market Intelligence
Automates research on industry trends, competition, and regulatory factors.
Provides real-time validation of startup claims, reducing reliance on manual verification.
Enhances decision-making with dynamic scoring models that adapt to market conditions.
Strategic Investment Decision Support
Identifies promising outliers that traditional screening might miss.
Improves portfolio diversification through objective opportunity assessment.
Enhances decision consistency across investment team members.
Increases deal flow quality through better founder targeting and feedback.

    """

    # Sample lead details
    sample_lead = {
        'Name': 'Rohit Jain',
        'Lead_ID': 1010101,
        'Current experience': """CoinDCX
CoinDCX
Full-time ¬∑ 4 yrs 4 mos
Full-time ¬∑ 4 yrs 4 mos
Head Web3 and DeFi (Okto)
Head Web3 and DeFi (Okto)
2024 - Present ¬∑ 1 yr 4 mos
2024 to Present ¬∑ 1 yr 4 mos
MD, CoinDCX Ventures
MD, CoinDCX Ventures
2021 - Present ¬∑ 4 yrs 4 mos
2021 to Present ¬∑ 4 yrs 4 mos
Chief Strategy and Investment Officer
Chief Strategy and Investment Officer
2021 - Present ¬∑ 4 yrs 4 mos
2021 to Present ¬∑ 4 yrs 4 mos
Bengaluru, Karnataka, India
Bengaluru, Karnataka, India
Head Web3 and DeFi (Okto)
Head Web3 and DeFi (Okto)
2024 - Present ¬∑ 1 yr 4 mos
2024 to Present ¬∑ 1 yr 4 mos
Angel portfolio:
ASQI (secured lending platform on blockchain; securitized by tokenized traditional financial assets like stocks, bonds)
Ava Labs (blockchain/smartcontract platform)
RIA (digital insurer in India)
TheList (global luxury ecommerce)
Canvas (HR Tech focused on diversity hires)

Advisor/Mentor:
CV Labs (global blockchain accelerator)
BuildersTribe (India blockchain accelerator)
100x.vc
TheThirdPillar (Upwork on Blockchain)
Goals101 (FIntech - API based solutions for banks)
Angel portfolio:
ASQI (secured lending platform on blockchain; securitized by tokenized traditional financial assets like stocks, bonds)
Ava Labs (blockchain/smartcontract platform)
RIA (digital insurer in India)
TheList (global luxury ecommerce)
Canvas (HR Tech focused on diversity hires)

Advisor/Mentor:
CV Labs (global blockchain accelerator)
BuildersTribe (India blockchain accelerator)
100x.vc
TheThirdPillar (Upwork on Blockchain)
Goals101 (FIntech - API based solutions for banks)
Fundamentum is a $100M growth stage fund, founded by Nandan Nilekani and Sanjeev Aggarwal.

Portfolio:
Travel Triangle
PharmEasy
Spinny
Fareye
Fundamentum is a $100M growth stage fund, founded by Nandan Nilekani and Sanjeev Aggarwal.

Portfolio:
Travel Triangle
PharmEasy
Spinny
Fareye
Helping and mentoring tech entrepreneurs with fund-raising, strategy and product development. Evaluating opportunities esp in the B2B e-commerce space.

Launched Buyoco - a B2B Crossborder E-Commerce platform that helps retailers in India (to begin with) import from manufacturers in China and Bangladesh (to begin with) and give them an end-to-end fulfilment experience.
Helping and mentoring tech entrepreneurs with fund-raising, strategy and product development. Evaluating opportunities esp in the B2B e-commerce space.

Launched Buyoco - a B2B Crossborder E-Commerce platform that helps retailers in India (to begin with) import from manufacturers in China and Bangladesh (to begin with) and give them an end-to-end fulfilment experience.'""",
        'Education': """Education
Education
Harvard Business School
Harvard Business School
MBA
MBA
2004 - 2006
2004 - 2006
Indian Institute of Technology, Guwahati
Indian Institute of Technology, Guwahati
Bachelor of Technology (BTech), Computer Science
Bachelor of Technology (BTech), Computer Science
1997 - 2001
1997 - 2001
Activities and societies: Cultural Secretary (Gymkhana Council), Organizer Alcheringa (annual Cultural festival), Co-organizer Techniche (annual Technical festival), Captain - Soccer Team, Member - Table Tennis and Athletics teams; Organizer Dramatics Club and other clubs on-campus.
Show all 3 educations""",
        'Company': 'CoinDCX',
        'Company overview': """Established in 2018, CoinDCX is the preferred crypto exchange in India, but also an instrumental player in building the broader Web3 ecosystem. 

Trusted by over 1.4 crore registered users. Our mission is simple: to provide easy access to Web3 experiences and democratize investments in virtual digital assets. We prioritize user safety and security, strictly adhering to KYC and AML guidelines. 

In our commitment to quality, we employ a stringent 7M Framework for the listing of crypto projects, ensuring users access only the safest virtual digital assets. CoinDCX has partnered with Okto for India to launch a secure multi-chain DeFi app that offers a keyless, self-custody wallet . It aims to simplify the world of decentralized finance (DeFi) by providing a secure, user-friendly, and innovative solution for managing virtual digital assets. 

Through CoinDCX Ventures, we've invested in over 15 innovative Web3 projects, reinforcing our dedication to the Web3 ecosystem. Our flagship educational initiative, #NamasteWeb3, empowers Indians with web3 knowledge, preparing them for the future of virtual digital assets. CoinDCX's vision and potential have gained the confidence of global investors, including Pantera, Steadview Capital, Kingsway, Polychain Capital, B Capital Group, Bain Capital Ventures, Cadenza, Draper Dragon, Republic, Kindred, and Coinbase Ventures. 

At CoinDCX, we're leading India towards the decentralized future of Web3 with an unwavering commitment to safety, simplicity, and education.""",
        'Company industry': 'Financial Services'
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

    print(generate_email_for_single_lead(sample_lead, product_details))
    # print(generate_email_for_multiple_leads(sample_leads, product_details))


if __name__ == "__main__":
    main()


