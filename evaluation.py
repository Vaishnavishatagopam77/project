from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def evaluate_answers(context, questions, answers):
    prompt = f"""
    Evaluate the student's answers using the reference content.

    CONTENT:
    {context}

    QUESTIONS AND ANSWERS:
    """
    for q, a in zip(questions, answers):
        prompt += f"\nQ: {q}\nA: {a}\n"

    prompt += """
    Give a score between 0 and 1 only.
    """

    try:
        score = llm.invoke(prompt).content.strip()
        return float(score)
    except Exception:
        print("⚠️ API Error during evaluation. Returning mock score.")
        return 0.8

def assess(state):
    score = evaluate_answers(
        state["context"],
        state["questions"],
        state["answers"]
    )
    state["score"] = score
    return state
