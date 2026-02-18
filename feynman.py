try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if ChatOpenAI:
    llm = ChatOpenAI(model="gpt-4o-mini")
else:
    llm = None

def feynman_explain(topic, context):
    prompt = f"""
    The student did not understand this topic.

    Reference content:
    {context}

    Re-explain "{topic}" in very simple terms,
    like teaching a 12-year-old.
    Use analogies and simple examples.
    """
    try:
        if not llm: raise Exception("LLM not available")
        return llm.invoke(prompt).content
    except Exception as e:
        print(f"Feynman Error: {e}")
        return f"Simulated Feynman Explanation for {topic}: Imagine this concept as a simple everyday analogy..."

def feynman_explain_batch(topics, context):
    """
    Explains multiple topics in one go to save time and tokens.
    Returns a dictionary {topic: explanation}.
    """
    topics_str = ", ".join(topics)
    prompt = f"""
    The student needs simple, Feynman-style explanations for the following concepts:
    {topics_str}

    Reference content:
    {context}

    Task:
    For EACH concept, provide a simple, beginner-friendly explanation (approx 2-3 sentences) using analogies.
    
    Output Format (JSON):
    {{
        "Concept Name 1": "Explanation...",
        "Concept Name 2": "Explanation...",
        ...
    }}
    """
    try:
        if not llm: raise Exception("LLM not available")
        response = llm.invoke(prompt).content
        import json
        
        # Clean up code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
            
        return json.loads(response)
    except Exception as e:
        print(f"Batch Feynman Error: {e}")
        
        # Hardcoded fallback explanations for AI concepts if the LLM fails
        fallbacks = {
            "Intelligent Agent": "An intelligent agent is a smart program that learns and makes decisions on its own. It perceives its environment through sensors and takes actions to reach a goal, much like a smart thermostat or a self-driving car.",
            "A* Search & Heuristics": "A* Search is like using a GPS. Heuristics are the 'shortcuts' or educated guesses that help it find the best path faster by focusing on the most promising directions.",
            "Knowledge Representation": "This is how an AI stores facts and rules. Think of it like a very organized filing cabinet that the AI uses to look up information and make logical decisions.",
            "Minimax Algorithm": "Minimax is used for games like Chess. It looks ahead to minimize the possible loss while maximizing the gain, assuming the opponent is also playing as well as they can.",
            "Perception in AI": "Perception is how an AI 'sees' or 'hears' the world. It uses cameras (eyes) and microphones (ears) to turn raw signals into data it can understand.",
            "History of AI (McCarthy)": "John McCarthy is known as the 'father of AI.' He officially gave artificial intelligence its name way back in 1956 at a famous conference.",
            "Logic Programming": "Logic programming is like giving the AI a set of rules and facts, and then asking it to solve a puzzle. It uses logic to find answers based on what it knows is true.",
            "Planning in AI": "Planning is when an AI creates a step-by-step to-do list to reach a goal. Instead of just reacting, it thinks ahead about the best sequence of moves.",
            "Strong AI": "Strong AI, or AGI, is the idea of a machine that can think and learn just like a human. It wouldn't just be good at one thing; it could understand any task."
        }
        

        # Improved dynamic fallback: Extract sentences from context containing the topic
        results = {}
        for t in topics:
            if t in fallbacks:
                results[t] = fallbacks[t]
            else:
                # Basic extractive summarization
                import re
                explanation = []
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
                count = 0
                for sent in sentences:
                    if t.lower() in sent.lower():
                        explanation.append(sent.strip())
                        count += 1
                        if count >= 2: break
                
                if explanation:
                    results[t] = " ".join(explanation)
                else:
                    results[t] = f"The concept of '{t}' is a core component of this topic. Please review the main lesson text for its specific definition and usage within this context."
        
        return results

