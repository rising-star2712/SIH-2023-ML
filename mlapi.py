from fastapi import FastAPI
from pydantic import BaseModel
from app import index , preprocess_input , 
app = FastAPI()

class ScoringPoint(BaseModel):
    

@app.post('/')
async def scoring_endpoint(item : ScoringPoint):
    #print(item.symptoms)
    return predictDisease(item.symptoms)