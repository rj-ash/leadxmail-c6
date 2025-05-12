from dotenv import load_dotenv
import os
import warnings
import io
from langchain_openai import ChatOpenAI
import time
from pydantic import BaseModel, Field
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

class EmailResponse(BaseModel):
    subject: str
    body: str
    lead_id: str

# Define the email generation prompt
prompt = """You are an B2B expert marketer. Based on the list of leads, their details and the product document provides , write a personalized email to the lead  in a concise manner, keeping in mind they have limited time to read the email. Write in a way that a fifth year student can understand and state your objective of helping the lead with your product"""
style = """
START THE EMAIL WITH A HEADLINE (ALL CAPS, STATE PURPOSE OF THE PRODUCT IN ONE LINE, E.G. WE WANT TO AUTOMATE YOUR ENTIRE HIRING PROCESS (DEPENDING ON THE PRODUCT), YOU FOCUS ON GROWTH. LET THE AI HANDLE (THE MANUAL WORK THAT WE ARE AUTOMATING WITH OUR PRODUCT), Your Process, Reimagined with AI(add personal touch), We Design AI That Gets the Job Done—Without You Lifting a Finger(add personal touch))
Do not use any common templates like I hope this email finds you well, etc.
Do not use too much buttering and unnecessary words.
Be casual, friendly and direct. Use human like tone and language. Follow this style:
1. First Person Pronouns: I, me, my, mine, we, us, our, ours. Write as if you are the one talking to the lead and do not use generalised statements.
2. Fillers & Disfluencies
Spoken or informal written human language often includes:
	•	uh, um, like, you know, kinda, sorta, actually, basically, literally
	•	contractions: gonna, wanna, gotta, ain't
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
            - name: str
            - lead_id: str
            - experience: str
            - education: str
            - company: str
            - company_overview: str
            - company_industry: str
        product_details (str): Product documentation/information
        
    Returns:
        dict: Dictionary containing 'subject', 'body', and 'lead_id' of the email
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        
    model = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.9,
        max_tokens=4096,
    )
        
    # Construct the prompt with lead details and product information
    instructed_prompt = f"""
Based on the following lead details, product information and the prompt, generate a personalized email and follow the personalisation convention for the lead.
prompt: ```{prompt}```
style: ```{style}```
```It shoud be casual and not too formal. Avoide using unnecessary words and too much buttering. Be casual, friendly and direct. Each email should have:  1. **Subject:** The subject line should be catchy, highly personalized and relevant to the lead's profile.Style: Frame it like a question or a catchy statement that arises curiosity in the reader's mind. (“[First Name], have you already solved this?”, “Is this what [Company Name] is missing?”, “Mind if I share something unusual?(add personal touch)”,“We noticed something strange in [industry]”, “One simple shift = big results”(add personal touch), “This made me pause—thought of your team”(add personal touch)) 2. **Body:** Start with telling the leads that you were going through your profile and talk a bit about his company. Then in next para, talk about problems in the industry of his company and introduce your product stating how it solve this problem. Get the details from product_details section. Then a compelling CTA asking him to schedule a meet or a call to discuss it further. Keep it short direct, casual, catchy and dont use too much formal jargonand avoid common templates (like I hope this email finds you well, etc). It should look as it is written by a human after carefully studying his profile and his company's details.```
Lead Details: {lead_details}
Product Information: {product_details}
Generate a personalized email following the format specified in the prompt above.
Return the results in a dictionary with these keys:
1. 'subject': The email subject line (style as provided)
2. 'body': The email body content
3. 'lead_id': The lead ID
Make sure the email is unique and highly personalized based on the lead's profile and relevant product features.
"""
    
    # Structure the output in given pydantic format
    structured_email = model.with_structured_output(EmailResponse)

    # Invoke the model to generate the email
    email_response = structured_email.invoke(instructed_prompt)

    # Convert the email response to a dictionary    
    email_response_dict = email_response.model_dump()

    # Return the email response
    return email_response_dict


def generate_email_for_multiple_leads(leads_list: list, product_details: str) -> list:
    """
    Generate personalized emails for multiple leads
    
    Args:
        leads_list (list): List of dictionaries, where each dictionary contains lead details with keys:
            - name: str
            - lead_id: str
            - experience: str
            - education: str
            - company: str
            - company_overview: str
            - company_industry: str
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
                print(f"Error processing lead {lead.get('name', 'Unknown')}: {str(e)}")
                batch_emails.append({
                    'subject': 'Error generating email',
                    'body': f'Error generating personalized email: {str(e)}',
                    'lead_id': str(lead.get('lead_id'))
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
        'name': 'Rohit Jain',
        'lead_id': 1010101,
        'experience': """CoinDCX
Angel portfolio:
ASQI (secured lending platform on blockchain; securitized by tokenized traditional financial assets like stocks, bonds) Ava Labs (blockchain/smartcontract platform) RIA (digital insurer in India) TheList (global luxury ecommerce) Canvas (HR Tech focused on diversity hires)
Advisor/Mentor:
CV Labs (global blockchain accelerator)
BuildersTribe (India blockchain accelerator)
100x.vc
TheThirdPillar (Upwork on Blockchain)
Goals101 (FIntech - API based solutions for banks)
Fundamentum is a $100M growth stage fund, founded by Nandan Nilekani and Sanjeev Aggarwal.

Portfolio: Travel Triangle PharmEasy Spinny Fareye
Helping and mentoring tech entrepreneurs with fund-raising, strategy and product development. Evaluating opportunities esp in the B2B e-commerce space.

Launched Buyoco - a B2B Crossborder E-Commerce platform that helps retailers in India (to begin with) import from manufacturers in China and Bangladesh (to begin with) and give them an end-to-end fulfilment experience.'""",
        'Education': """Education
Harvard Business School
MBA
2004 - 2006
Indian Institute of Technology, Guwahati    
Bachelor of Technology (BTech), Computer Science
1997 - 2001
Activities and societies: Cultural Secretary (Gymkhana Council), Organizer Alcheringa (annual Cultural festival), Co-organizer Techniche (annual Technical festival), Captain - Soccer Team, Member - Table Tennis and Athletics teams; Organizer Dramatics Club and other clubs on-campus.
Show all 3 educations""",
        'company': 'CoinDCX',
        'company_overview': """Established in 2018, CoinDCX is the preferred crypto exchange in India, but also an instrumental player in building the broader Web3 ecosystem. 

Trusted by over 1.4 crore registered users. Our mission is simple: to provide easy access to Web3 experiences and democratize investments in virtual digital assets. We prioritize user safety and security, strictly adhering to KYC and AML guidelines. 

In our commitment to quality, we employ a stringent 7M Framework for the listing of crypto projects, ensuring users access only the safest virtual digital assets. CoinDCX has partnered with Okto for India to launch a secure multi-chain DeFi app that offers a keyless, self-custody wallet . It aims to simplify the world of decentralized finance (DeFi) by providing a secure, user-friendly, and innovative solution for managing virtual digital assets. 

Through CoinDCX Ventures, we've invested in over 15 innovative Web3 projects, reinforcing our dedication to the Web3 ecosystem. Our flagship educational initiative, #NamasteWeb3, empowers Indians with web3 knowledge, preparing them for the future of virtual digital assets. CoinDCX's vision and potential have gained the confidence of global investors, including Pantera, Steadview Capital, Kingsway, Polychain Capital, B Capital Group, Bain Capital Ventures, Cadenza, Draper Dragon, Republic, Kindred, and Coinbase Ventures. 

At CoinDCX, we're leading India towards the decentralized future of Web3 with an unwavering commitment to safety, simplicity, and education.""",
        'Company industry': 'Financial Services'
    }

    

    single_email = generate_email_for_single_lead(sample_lead, product_details)
    print(single_email)
    print(type(single_email))


if __name__ == "__main__":
    main()


