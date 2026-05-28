from pydantic import BaseModel

class ModelRequest(BaseModel):
    email: str
