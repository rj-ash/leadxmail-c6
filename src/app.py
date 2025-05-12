from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
from typing import List, Optional, Union
from personalised_email import generate_email_for_single_lead, generate_email_for_multiple_leads

app = FastAPI(
    title="Personalized Email Generation API",
    description="API for generating personalized emails based on lead and product information",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your frontend origin
    allow_credentials=False,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class LeadDetails(BaseModel):
    name: str
    lead_id: str
    experience: str
    education: str
    company: str
    company_overview: str
    company_industry: str

class EmailResponse(BaseModel):
    lead_id: str
    subject: str
    body: str

class ProductDetails(BaseModel):
    details: str

@app.post("/generate-single-email", response_model=dict)
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
        # Convert Pydantic model to dict using model_dump()
        lead_dict = lead.model_dump()
        # Generate email
        result = generate_email_for_single_lead(lead_dict, product.details)
        # Ensure lead_id is included in the response
        if 'lead_id' not in result or result['lead_id'] is None:
            result['lead_id'] = lead.lead_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating email: {str(e)}")

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
        # Convert Pydantic models to dicts using model_dump()
        leads_dict = [lead.model_dump() for lead in leads]
        # Generate emails
        results = generate_email_for_multiple_leads(leads_dict, product.details)
        # Ensure lead_id is included in each response
        for i, result in enumerate(results):
            if 'lead_id' not in result or result['lead_id'] is None:
                result['lead_id'] = leads[i].lead_id
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