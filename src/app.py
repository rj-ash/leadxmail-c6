from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
from typing import List, Optional
from personalised_email import generate_email_for_single_lead, generate_email_for_multiple_leads

app = FastAPI(
    title="Personalized Email Generation API",
    description="API for generating personalized emails based on lead and product information",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://flow-forge-campaigns.lovable.app"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class LeadDetails(BaseModel):
    name: str
    lead_id: int
    experience: str
    education: str
    company: str
    company_overview: str
    company_industry: str

class EmailResponse(BaseModel):
    lead_id: int
    subject: str
    body: str

class ProductDetails(BaseModel):
    details: str

@app.post("/generate-single-email", response_model=EmailResponse)
async def generate_single_email(lead: LeadDetails, product: ProductDetails):
    """
    Generate a personalized email for a single lead
    
    Args:
        lead: Lead details including name, experience, education, etc.
        product: Product information and details
        
    Returns:
        Dictionary containing subject and body of the generated email
    """
    try:
        # Convert Pydantic model to dict
        lead_dict = lead.dict()
        # Generate email
        result = generate_email_for_single_lead(lead_dict, product.details)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-multiple-emails", response_model=List[EmailResponse])
async def generate_multiple_emails(leads: List[LeadDetails], product: ProductDetails):
    """
    Generate personalized emails for multiple leads
    
    Args:
        leads: List of lead details
        product: Product information and details
        
    Returns:
        List of dictionaries, each containing subject and body of generated emails
    """
    try:
        # Convert Pydantic models to dicts
        leads_dict = [lead.dict() for lead in leads]
        # Generate emails
        results = generate_email_for_multiple_leads(leads_dict, product.details)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 