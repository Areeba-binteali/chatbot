from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents import Runner
from agent import agent  # 👈 existing agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ya frontend URL like "http://localhost:3000"
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=QueryResponse)
async def ask_agent(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        # ✅ async call
        result = await Runner.run(
            agent,
            input=req.question
        )

        return QueryResponse(
            answer=result.final_output
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
