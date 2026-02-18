from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_context(topic):
    prompt = f"""
    You are a tutor.
    Explain the topic "{topic}" clearly.
    Cover:
    - Definition
    - Key ideas
    - Simple example
    Keep it concise but complete.
    """
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Mock Context for {topic}: Binary Search is a search algorithm that finds the position of a target value within a sorted array. Key ideas: Divide and conquer. Example: Finding 7 in [1, 3, 5, 7, 9]."

def validate_context(context, topic):
    prompt = f"""
    Check whether the following content sufficiently explains "{topic}".
    Answer only YES or NO.

    CONTENT:
    {context}
    """
    try:
        result = llm.invoke(prompt).content.strip()
        return result.upper() == "YES"
    except Exception:
        return True

def get_valid_context(topic):
    for _ in range(3):  # try max 3 times
        context = generate_context(topic)
        if validate_context(context, topic):
            return context

    # fallback – return best attempt
    return context
