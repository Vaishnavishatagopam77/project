from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_questions(state):
    prompt = f"""
    Based ONLY on the following content, generate 3 questions.

    CONTENT:
    {state['context']}
    """
    try:
        questions = llm.invoke(prompt).content.split("\n")
    except Exception as e:
        print(f"⚠️ API Error: {e}. Using mock questions.")
        questions = ["What is the time complexity of Binary Search?", "Does the array need to be sorted?", "What is the middle element relation?"]
    
    state["questions"] = [q for q in questions if q.strip()]
    return state



from typing import TypedDict, List
from langgraph.graph import StateGraph
from evaluation import evaluate_answers
from feynman import feynman_explain

class State(TypedDict):
    topic: str
    context: str
    questions: List[str]
    answers: List[str]
    score: float

def assess(state):
    score = evaluate_answers(state["context"], state["questions"], state["answers"])
    state["score"] = score
    return state
def route(state):
    if state["score"] >= 0.7:
        return "pass"
    else:
        return "fail"
builder = StateGraph(State)

builder.add_node("questions", generate_questions)
builder.add_node("assess", assess)

builder.add_conditional_edges(
    "assess",
    route,
    {
        "pass": "__end__",
        "fail": "__end__"
    }
)


builder.set_entry_point("questions")
graph = builder.compile()
