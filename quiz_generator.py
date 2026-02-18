import os
import json
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def generate_quiz_mock(num_questions=5):
    """
    Returns mock questions when API is unavailable.
    """
    print("⚠️ API Unavailable (Quota/Connection). Using MOCK questions.")
    return [
        {
            "q": f"Mock Question {i+1} from generated content?",
            "opts": ["Option A", "Option B", "Option C", "Option D"],
            "a": 1,
            "c": "mock_concept",
            "e": "This is a mock explanation because the API could not be reached."
        }
        for i in range(num_questions)
    ]

def generate_quiz(content, num_questions=5, question_type="MCQs"):
    """
    Generates a quiz based on the provided content using the specified rules.
    Returns a list of dictionaries formatting suitable for the learning system.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY not found in environment.")
        return generate_quiz_mock(num_questions)

    if not ChatOpenAI:
        print("⚠️ ChatOpenAI lib not found.")
        return generate_quiz_mock(num_questions)

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    except Exception as e:
        print(f"⚠️ Error initializing LLM: {e}")
        return generate_quiz_mock(num_questions)
    
    # We enforce JSON format in the system prompt to ensure parsability
    prompt = f"""
    You are a content-bound question generator.

    Rules:
    1. Generate questions ONLY from the given content.
    2. Do NOT use outside knowledge or assumptions.
    3. Each question must have a clear answer within the content.
    4. If information is not present, do not ask about it.
    5. Do not rephrase or extend ideas beyond the text.
    6. Evaluate the relevance of each question to the core concepts (0-10).

    Content:
    {content}

    Task:
    Generate {num_questions} {question_type} strictly based on the above content.
    
    Output Format:
    Provide the output as a valid JSON object with a single key "questions" containing a list of question objects.
    Each question object must have the following structure:
    {{
        "q": "The question text",
        "opts": ["Option 1", "Option 2", "Option 3", "Option 4"],
        "a": integer_correct_option_index_1_to_4,
        "c": "concept_category_one_word",
        "e": "Brief explanation referencing the content",
        "s": integer_relevance_score_0_to_10
    }}
    Ensure "a" is an integer between 1 and 4.
    """

    try:
        response = llm.invoke(prompt).content
        
        # Clean up code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
            
        data = json.loads(response)
        questions = data.get("questions", [])
        
        # Filter by relevance score (keep only high quality questions)
        valid_questions = [q for q in questions if q.get('s', 0) >= 7]
        
        # If we filtered too many, just return the top ones, or all if none met the threshold (fallback)
        if not valid_questions and questions:
            return questions
            
        return valid_questions
        
    except Exception as e:
        print(f"⚠️ Error generating quiz: {e}")
        return generate_quiz_mock(num_questions)


def generate_reassessment_quiz(context, previous_questions, num_questions=10):
    """
    Generates a reassessment quiz ensuring no overlap with previous questions.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OPENAI_API_KEY not found. Returning empty list to trigger static fallback.")
        return []

    if not ChatOpenAI:
        return []

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    except Exception as e:
        print(f"⚠️ Error initializing LLM: {e}")
        return []

    # Format previous questions for the prompt
    prev_q_text = "\n".join([f"- {q}" for q in previous_questions])

    prompt = f"""
    You are an AI assessment generator with strict anti-repetition rules.
    
    Context:
    This is a REASSESSMENT quiz. Some questions have already been asked before.
    
    Rules (MANDATORY):
    1. Do NOT repeat any previously asked question listed below.
    2. Do NOT rephrase, paraphrase, or slightly modify old questions.
    3. Do NOT test the same concept in the same way again.
    4. Each question must assess a DIFFERENT concept or a clearly different angle.
    5. Before generating each question, internally check:
       - Has this question or concept appeared before?
       - If yes → discard and generate a new one.
    6. Perform a final validation step:
       - Confirm that all 5 questions are unique in wording AND intent.
    
    Previous Questions (AVOID THESE):
    {prev_q_text}
    
    Content:
    {context}
    
    Output Format (JSON):
    {{
        "questions": [
            {{
                "q": "Question text",
                "opts": ["A", "B", "C", "D"],
                "a": 1, # Correct index 1-4
                "c": "concept_tag",
                "e": "Explanation of why correct answer is right and others are wrong.",
                "s": integer_relevance_score_0_to_10
            }}
        ]
    }}
    
    If unique questions cannot be generated, return exactly:
    {{
        "error": "Unable to generate non-repetitive questions within the given constraints."
    }}
    """

    try:
        response = llm.invoke(prompt).content
        
        # Clean up code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
            
        data = json.loads(response)
        
        # Check for error signal
        if "error" in data:
            print(f"⚠️ Generator reported: {data['error']}")
            return []
            
        questions = data.get("questions", [])
        
        # Filter by relevance score (keep only high quality questions)
        valid_questions = [q for q in questions if q.get('s', 0) >= 7]
        
        if not valid_questions and questions:
             return questions
             
        return valid_questions
        
    except Exception as e:
        #print(f"⚠️ Error generating reassessment: {e}")
        return []
