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
style = """
Start with a greeting and then go on to say that you are reaching out because of [reason].
Do not use too much buttering and unnecessary words specially after the first line (I was going through your profile and noticed your inspiring journey is phenomenal,fabulous, etc. Don't use such unnecessary buttering. Jus go like "I was going through your profile and noticed [mention things which are relevant to product and matches with the lead's profile, no unnecessary buttering]")
Be casual, friendly and direct. 
It should be a short, personalized email (100–150 words) to a potential lead who could benefit from your product/service/solution. It should:
                    1.	Start with deep personalization — Reference something specific about the lead’s business, role, recent announcement, or pain point you’ve identified. Show you understand them—not just their company name.
                    2.	Make a relevant connection — Briefly explain who you are and why you’re reaching out. Make it clear why they specifically are a fit for what you offer.
                    3. Talk about the problems in the lead's industry and how the product can solve the problem.
                    4.	Focus on value (not features) — Position your solution around a problem or opportunity that matters to them. Avoid a hard sell—offer insight, benefit, or a useful idea that shows you can help.
                    5.	Keep it short and natural — Write like a human, not a sales robot. End with a simple CTA (e.g., “open to a quick chat?” or “would you be interested in exploring this further?”).

Use human like tone and language. Follow this style:
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

product_database = """
parchaa = {
    "problem_it_solves": (
        "Long OPD wait times due to high patient footfall; "
        "short consultation durations and overworked staff; "
        "manual processes and lack of automation; "
        "poor patient engagement and adherence; "
        "lack of authentic health data for governance and research; "
        "insufficient infrastructure and digital tools in rural/NGO settings; "
        "inefficiencies and errors in emergency room triaging; "
        "limited mental healthcare access in academic institutions."
    ),
    "solution": (
        "A comprehensive AI-powered digital health platform covering OPD, emergency, air-gapped, and wellness workflows. "
        "It supports the full healthcare journey—onboarding, consultation, follow-up, and monitoring—while being interoperable "
        "with existing hospital systems and government initiatives like ABHA and Ayushman Bharat. "
        "It offers remote, offline-compatible deployments, smart dashboards, and CDSS support."
    ),
    "unique_selling_point": (
        "End-to-end digital healthcare platform functional in both online and offline modes; "
        "powered by advanced AI/LLM models trained on Indian and global clinical standards (ICMR, WHO, CDC, etc.); "
        "first-in-class emergency room automation tools including triage classification and resource dashboards; "
        "deep focus on underserved areas and community care; "
        "modular, white-label deployment suitable for hospitals, NGOs, and academic institutions."
    ),
    "features": (
        "Smart OPD with AI-led triaging, diagnosis, and prescriptions; "
        "Clinical Decision Support System (CDSS) with alerts for ADRs and contra-indications; "
        "Patient app with teleconsultation, medication reminders, and adherence tracking; "
        "Emergency module with auto-triaging and saturation dashboards; "
        "Air-gapped mode for rural deployment with vitals tracking, inventory management, and follow-up tools; "
        "WellKiwi for campus wellness including mental health support, insurance, and health analytics; "
        "Dashboards for administrators, wardens, and doctors; "
        "Interoperability with existing HIS, ABHA, PHRs, pharmacies, and diagnostics (e.g., 1mg integration)."
    ),
    "benefits": (
        "Reduces patient wait time and improves quality of consultations; "
        "supports doctors with AI-based decision tools; "
        "boosts hospital efficiency through automation and data insights; "
        "enables healthcare access in rural and remote regions; "
        "improves patient follow-up, engagement, and outcomes; "
        "promotes mental wellness and proactive health management in students; "
        "enables fast, accurate emergency triaging; "
        "standardizes care protocols and improves collaboration among stakeholders."
    )
}
,
"predCo": {
  "problem_it_solves": [
    "Unplanned equipment failures causing production delays and financial losses.",
    "Inefficient maintenance strategies leading to safety concerns and operational disruptions.",
    "Fragmented systems and lack of centralized monitoring in industrial operations.",
    "Lack of real-time visibility and decision-making across assets and departments.",
    "Stockouts, overstocking, and asset mismanagement in supply chain operations.",
    "Manual, error-prone inventory tracking methods.",
    "Delayed threat detection and response in surveillance operations."
  ],
  "solution": [
    "AI-powered predictive maintenance and condition monitoring platform.",
    "Digital Twin technology to simulate and optimize operations in real time.",
    "Smart inventory management using RFID and computer vision.",
    "Centralized monitoring and data integration for legacy systems and SCADA.",
    "GenAI-powered assistants for real-time document retrieval and support.",
    "Geofencing solutions for real-time asset tracking and alerts.",
    "Role-based access and proactive alert systems for operations control."
  ],
  "unique_selling_point": [
    "Unified AI-driven platform integrating IoT, ML, and digital twin technologies.",
    "Real-time actionable insights across a wide array of machinery and assets.",
    "Highly scalable and adaptable solutions tailored for legacy and modern systems.",
    "Seamless integration with existing infrastructure, no new hardware needed.",
    "Use-case agnostic architecture spanning energy, manufacturing, logistics, and security."
  ],
  "features": [
    "Dynamic dashboards with real-time visualization and KPI tracking.",
    "Customizable alerts based on threshold breaches and anomaly detection.",
    "Real-time data acquisition from diverse IoT sensors.",
    "Integrated ML models to predict equipment failure and remaining useful life.",
    "Computer vision for automated shelf and item recognition.",
    "GenAI tools for document digitization and knowledge retrieval.",
    "Geofencing alerts and asset movement tracking.",
    "Role-based access and operational rule configuration."
  ],
  "benefits": [
    "Reduce equipment downtime by 35-40%.",
    "Extend asset lifespan by up to 30%.",
    "Lower maintenance costs by 8-12%.",
    "Improve inventory turnover by 40% and reduce shrinkage by 50%.",
    "Minimize manual monitoring efforts and decision-making delays.",
    "Achieve up to 87% accuracy in lifecycle predictions of critical components.",
    "Enhance workplace safety and compliance through AI-powered surveillance.",
    "Boost operational transparency and inter-departmental collaboration."
  ]
},
"InvestorBase": {
    "problem_it_solves": [
        "Overwhelming volume of inbound pitch decks makes it hard for VCs to evaluate each one thoroughly.",
        "Current deal evaluation processes are manual, slow, biased, and inconsistent.",
        "Due diligence is time-consuming, taking up to 2 weeks per deal.",
        "Missed opportunities due to delays and inefficient workflows."
    ],
    "solution": [
        "AI-driven pitch deck analyzer that extracts key information instantly.",
        "Human-augmented analysis ensures depth and credibility of insights.",
        "Dynamic opportunity scoring aligned with the fund’s investment thesis.",
        "Automated red flag detection, validation, and memo generation."
    ],
    "unique_selling_point": [
        "Combines speed and precision of AI with expert analyst judgment.",
        "Customizable scoring tailored to specific investment theses.",
        "Scalable solution that handles 10 to 1,000+ decks per month.",
        "Delivers analyst-grade insights in 24–48 hours."
    ],
    "features": [
        "Pitch Deck Analyzer – instant insights from uploaded decks.",
        "Thesis Alignment – auto scoring of decks based on fund criteria.",
        "InsightMaster – AI assistant for deeper analysis.",
        "Auto Analysis + Alerts – real-time notifications for matching deals.",
        "Market Intel – context-rich competitive and news insights.",
        "Investor Research – deeper, customized insights beyond basic data.",
        "Automated Collection – centralized collection from various sources.",
        "Investor Memos – auto-generated, ready-to-use investment memos."
    ],
    "benefits": [
        "Faster and smarter deal evaluation with reduced manual effort.",
        "Increased chances of discovering high-potential investments.",
        "Higher quality decisions through objective and consistent scoring.",
        "Significant time savings in screening and due diligence.",
        "Enhanced founder engagement and reduced deal drop-offs."
    ]
},
   "sankalpam": {
    "problem_it_solves": "Inefficient temple operations, outdated communication, inadequate resource management, limited accessibility for devotees, and challenges in preserving cultural heritage and managing religious tourism.",
    
    "solution": "A technology-driven platform that empowers temples through AI, IoT, and cloud tools to improve operational efficiency, enhance devotee engagement, enable secure fundraising, digitize cultural assets, and modernize communication.",
    
    "unique_selling_point": "Sankalpam bridges the gap between tradition and modernity, offering temples smart management tools, immersive devotee experiences, and government collaboration frameworks, all under one unified platform.",
    
    "features": [
        "AI-powered surveillance and crowd control",
        "IoT-enabled resource and environmental monitoring",
        "Mobile app for temple services, communication, and ticketing",
        "Digital donation platforms with global access",
        "AR/VR-based cultural immersion experiences",
        "Live streaming of religious rituals (Darshan)",
        "Virtual pooja booking (Sankalp)",
        "Online astrology consultations (Jyotish Vani)",
        "Sacred offering delivery (Prasadam)",
        "Pilgrimage planning assistance (Yatra)",
        "Comprehensive Hindu knowledge repository (Gyan Kosh)"
    ],
    
    "benefits": {
        "For Temples": [
            "Enhanced operational efficiency and crowd management",
            "Reduced administrative costs and better resource allocation",
            "Increased donations and new revenue streams",
            "Greater transparency in financial management",
            "Improved security and heritage preservation"
        ],
        "For Devotees": [
            "Seamless access to services through online bookings and virtual participation",
            "Personalized spiritual experiences",
            "Improved accessibility for elderly and differently-abled",
            "Interactive cultural education and deeper immersion"
        ],
        "For Governments": [
            "Smart heritage management using data and AI",
            "Boosted religious tourism and economic development",
            "Stronger cultural preservation and social impact through temple-centric community engagement"
        ]
    }
},
"Opticall": {
    "problem_it_solves": 
        "General: High call volumes, inconsistent call handling quality, delayed insights, and ineffective coaching that increase costs and reduce performance.\n"
        "Sales Center: Sales reps face chaotic lead volumes, cold leads, and lack real-time visibility into performance gaps, resulting in missed revenue opportunities.\n"
        "Contact Center: Overwhelmed agents, long wait times, inconsistent service, and buried insights lead to high costs, low customer satisfaction, and limited performance visibility.",
    
    "solution": 
        "General: A unified AI platform that automates queries via bots, supports agents in real time, and converts raw call data into actionable insights through customizable dashboards.\n"
        "Sales Center: Automates repetitive lead engagement, provides real-time pitch coaching, and surfaces deep post-call insights to optimize sales conversions.\n"
        "Contact Center: Automates customer queries with virtual agents, supports agents with real-time assistance, and delivers performance insights from every call to improve support quality.",
    
    "unique_selling_point": 
        "General: Modular, phygital-ready architecture with deep tech and lightweight deployment that fits any workflow and delivers real-time insights across audio, video, and text.\n"
        "Sales Center: Requires no change in existing sales workflows—adapts to your team with flexible tools, custom templates, and sales-specific dashboards.\n"
        "Contact Center: Fully customizable to existing support operations, delivering automation and real-time coaching without altering how your team works.",
    
    "features": 
        "General: Agent Assist, automated call scoring, dynamic dashboards, vernacular engine (28+ languages), real-time prompts, and video/audio analytics.\n"
        "Sales Center: Virtual sales agents, real-time objection handling, pitch prompts, AI-powered knowledge base, performance dashboards, multilingual support, and coaching tools.\n"
        "Contact Center: AI-powered virtual agents, real-time agent guidance with visual checklists, instant knowledge access, automated escalations, coaching hub, and compliance support.",
    
    "benefits": 
        "General: +18 CSAT points, 95% QA coverage (up from 2%), and 22% reduction in support costs with real-time insights and automation.\n"
        "Sales Center: 12% increase in conversions, 9% growth in monthly bookings, 18% reduction in customer acquisition cost, and accelerated deal closures.\n"
        "Contact Center: +18 CSAT points, 98% QA cove
,
    "IndikaAI": {
        "problem_it_solves": (
            "Enterprises face challenges in AI adoption due to unclear AI roadmaps, data readiness issues, "
            "lack of internal expertise, difficulty integrating AI with existing systems, and rapid technology changes. "
            "These factors hinder ROI and effective AI implementation."
        ),
        "solution": (
            "Indika AI provides end-to-end AI solutions, including strategy formulation, data preparation, "
            "foundation model selection, model fine-tuning, deployment, and continuous monitoring. It offers tailored "
            "products for industries like healthcare, judiciary, infrastructure, and customer service."
        ),
        "unique_selling_point": (
            "Comprehensive AI development lifecycle support, domain-specific AI products, access to 50,000+ experts "
            "across 100+ languages, and experience across sectors including judiciary, healthcare, infrastructure, and BFSI."
        ),
        "features": (
            "• AI strategy and roadmap development\n"
            "• Data digitization, anonymization, and labeling\n"
            "• Custom generative AI, NLP, computer vision, and audio processing\n"
            "• Platforms like DigiVerse (digitization), DataStudio (model training), FlexiBench (expert workforce)\n"
            "• Industry-specific AI solutions (Nyaay AI, PredCo, RoadVision AI, Parchaa AI, Choice AI)\n"
            "• Ready-to-deploy tools (OCR, speech-to-text, trust & safety, synthetic data generation)"
        ),
        "benefits": (
            "• Faster and smoother AI adoption\n"
            "• Improved operational efficiency and automation\n"
            "• Enhanced decision-making through AI-powered insights\n"
            "• Access to scalable AI solutions tailored to specific industries\n"
            "• Expert support across all AI development stages\n"
            "• Multilingual and multidisciplinary support"
        )
},
"Flexibench": {
    "problem_it_solves": "Lack of trained professionals with domain-specific expertise to contribute to AI systems effectively; need for culturally and linguistically aware AI development.",
    "solution": "A talent development hub that trains domain experts in hands-on AI tasks (e.g., data labeling, RLHF, annotation), offering real project experience and earnings while learning.",
    "unique_selling_point": "Combines real-world AI project exposure with specialized domain training and global language support, turning professionals into AI contributors.",
    "features": [
        "Training across 100+ languages for localization and bias reduction",
        "Tailored learning paths in fields like law, medicine, engineering, and linguistics",
        "Real industry experience through live AI projects",
        "Focus on essential AI lifecycle skills (data labeling, RLHF, annotation)",
        "Support for 20+ academic fields with 60,000+ experts onboarded"
    ],
    "benefits": [
        "Earn while gaining AI industry experience",
        "Transform domain knowledge into valuable AI contributions",
        "Enable accurate, culturally aware AI systems",
        "Promote India's global role in AI excellence",
        "Join a global network of trained professionals for AI projects"
    ]
},
"InspireAI": {
    "problem_it_solves": "Content creators and marketers face burnout, creative blocks, and inefficiencies when generating engaging and consistent content across platforms.",
    "solution": "InspireAI offers an AI-powered platform that generates personalized, high-quality, and engaging content at scale, reducing the creative workload and boosting productivity.",
    "unique_selling_point": "Combines personalization, scalability, and creative storytelling powered by AI, tailored specifically for content marketers and creators.",
    "features": [
        "AI-driven content generation",
        "Personalized content suggestions",
        "Multi-platform optimization",
        "Content calendar integration",
        "Real-time collaboration tools",
        "Performance analytics and insights"
    ],
    "benefits": [
        "Saves time and reduces creative fatigue",
        "Increases content output without compromising quality",
        "Boosts audience engagement through personalized storytelling",
        "Streamlines content workflow for teams and individuals",
        "Improves ROI through data-driven content strategies"
    ]
},
"Insituate": {
    "problem_it_solves": (
        "Businesses struggle with rapid AI advancements due to lack of AI talent, "
        "compliance concerns (handling sensitive data), and compatibility issues with legacy systems."
    ),
    "solution": (
        "A secure, no-code, end-to-end AI development platform enabling enterprises to adopt "
        "state-of-the-art LLMs and RAG pipelines, while keeping all data in-house."
    ),
    "unique_selling_point": (
        "The only platform to offer a no-code, one-stop solution for building end-to-end AI copilots "
        "that integrate with legacy systems and prioritize data security."
    ),
    "features": (
        "• In-situ (on-premise) database\n"
        "• No-code development interface\n"
        "• Ironclad security protocols\n"
        "• State-of-the-art LLM and RAG integration\n"
        "• 100+ pre-built templates\n"
        "• Compatibility with legacy software\n"
        "• Gridsearch for copilot optimization\n"
        "• Sentry mode for continuous improvement\n"
        "• AutoLLM deployment within a week\n"
        "• LLMOps capabilities\n"
        "• On-cloud and on-premise deployment options\n"
        "• Team collaboration and one-click export"
    ),
    "benefits": (
        "• Accelerates enterprise AI adoption without requiring in-house AI talent\n"
        "• Maintains data privacy and regulatory compliance\n"
        "• Saves time and cost (AutoLLM in 1 week vs. traditional 24 months)\n"
        "• Streamlines development and deployment with minimal technical barriers\n"
        "• Empowers internal teams to create domain-specific copilots\n"
        "• Taps into a large, global, multi-vertical market"
    )
},
"choiceAI": {
    "problem_it_solves": (
        "1. Lack of regulatory framework for OTT content certification.\n"
        "2. Inability of OTT platforms to filter offensive or harmful content effectively.\n"
        "3. Viewer concerns about exposure to explicit/inappropriate content and lack of parental controls.\n"
        "4. Production houses face delays and risks in content release due to censorship.\n"
        "5. Lack of personalized content access and inclusivity for diverse viewers."
    ),
    "solution": (
        "1. AI tool that ensures responsible content distribution and compliance.\n"
        "2. Personalized viewer experience using advanced tagging and filtering.\n"
        "3. Parental controls and content warnings to protect minors and sensitive viewers.\n"
        "4. Streamlined certification and approval process for content creators.\n"
        "5. Intelligent content assessment to maintain compliance without limiting creativity."
    ),
    "unique_selling_point": (
        "Only solution offering comprehensive customization, real-time AI moderation, "
        "personalized viewing, compliance with censorship, and collaboration with CBFC and OTT platforms."
    ),
    "features": (
        "1. Choice Tagger: Tags content (e.g., sex, violence, nudity) for CBFC and OTT.\n"
        "2. Choice Viewer: Lets users filter content based on tags.\n"
        "3. AI-powered moderation and certification.\n"
        "4. Personalized recommendations and curated experiences.\n"
        "5. Parental control and age-based filtering.\n"
        "6. Collaboration tools for CBFC and independent creators.\n"
        "7. OEM integration with OTT platforms."
    ),
    "benefits": (
        "1. Viewers enjoy safe, personalized, and culturally sensitive content.\n"
        "2. Parents can restrict inappropriate content for children.\n"
        "3. Content creators maintain creative freedom while complying with regulations.\n"
        "4. OTT platforms enhance user experience and reduce legal risk.\n"
        "5. CBFC streamlines verification and sets unified digital standards.\n"
        "6. Independent creators gain exposure and platform support.\n"
        "7. Faster content approval and reduced release delays."
    )
}
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
    messages = [
        {
            "role": "system",
            "content": (
                "You are an B2B expert marketer. Based on the list of leads, their details and the product document provides , write a personalized email to the lead  in a concise manner, keeping in mind they have limited time to read the email. Write in a way that a fifth year student can understand and state your objective of helping the lead with your product"""
                "You avoid formal templates, skip unnecessary flattery, and talk like a real person. Follow the tone and style guide provided."
                "Refer product database as per the given product details for context, problems in lead's industry and how the product can solve the problem, {product_database}"
                "Do fact check about the problems and after verifying the facts, put them in the email else dont."
            )
        },
        {
            "role": "user",
            "content": f"""
Task:
Write a personalized email for the lead above. Follow the subject/body formatting rules. Return the result in a JSON object with keys: 'subject', 'body', and 'lead_id'. Make sure the email must not contain any placeholders (like [xyz]). For sender's contact details, use the details given in the "ProductDetails" section.

Style Guide:
{style}

Lead Details:
{lead_details}

"ProductDetails":
{product_details}

Return the results in a dictionary with these keys:
1. 'subject': The email subject line (style as provided)
2. 'body': The email body content
3. 'lead_id': The lead ID

"""
        }
    ]
    try:
        api_key = os.getenv("OPENAI_API_KEY")
    except Exception as e:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        
    model = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.7,
        max_tokens=4096,
    )
        
    # Construct the prompt with lead details and product information
   

    
    # Structure the output in given pydantic format
    structured_email = model.with_structured_output(EmailResponse)

    # Invoke the model to generate the email
    email_response = structured_email.invoke(messages)

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


