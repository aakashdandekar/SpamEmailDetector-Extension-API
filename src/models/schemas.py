from pydantic import BaseModel

class Request(BaseModel):
    email: str