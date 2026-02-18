import random
import time
import sys

# Data Structures
TOPICS = {
    1: "Artificial Intelligence",
    2: "Machine Learning",
    3: "Deep Learning",
    4: "Data Science",
    5: "Computer Vision"
}

# --- Content Definitions ---
# Each topic has:
# - context: Academic explanation
# - questions: List of 10 initial questions with 'concept' tag
# - feynman: Dict mapping 'concept' to simple explanation
# - reassessment: Dict mapping 'concept' to list of extra questions

CONTENT = {
    1: { # AI
        "context": """
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.
These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction.
Particular applications of AI include expert systems, speech recognition, and machine vision.
        """,
        "questions": [
            {"q": "Who is considered the father of AI?", "opts": ["Alan Turing", "John McCarthy", "Elon Musk", "Charles Babbage"], "a": 2, "c": "history", "e": "John McCarthy coined the term in 1956.\nHe organized the Dartmouth Conference where AI was born."},
            {"q": "Which is NOT a type of AI?", "opts": ["Narrow AI", "General AI", "Super AI", "Magic AI"], "a": 4, "c": "types", "e": "Magic AI is not a scientific classification.\nAI types are based on capability: Narrow, General, Super."},
            {"q": "What is the Turing Test for?", "opts": ["Hardware speed", "Human-like intelligence", "Debugging", "Internet speed"], "a": 2, "c": "history", "e": "It tests if a machine's behavior is indistinguishable from a human.\nHardware speed is irrelevant to this test."},
            {"q": "Which field deals with human language?", "opts": ["CV", "NLP", "Robotics", "Cloud"], "a": 2, "c": "domains", "e": "NLP (Natural Language Processing) handles text and speech.\nCV handles images, Robotics handles movement."},
            {"q": "Deep Blue defeated the champion of which game?", "opts": ["Go", "Chess", "Jeopardy", "Poker"], "a": 2, "c": "history", "e": "Deep Blue defeated Garry Kasparov in Chess in 1997.\nAlphaGo is famous for defeating the Go champion."},
            {"q": "A* Search uses what function?", "opts": ["DFS", "BFS", "Heuristic", "Binary"], "a": 3, "c": "algos", "e": "A* uses a heuristic function f(n) = g(n) + h(n) to guide search.\nDFS and BFS do not use heuristics by default."},
            {"q": "What is an Expert System?", "opts": ["Learns from data", "Emulates human expert", "A robot", "A database"], "a": 2, "c": "domains", "e": "Expert Systems use rules to mimic human decision-making.\nThey do not necessarily 'learn' from data like ML models."},
            {"q": "Fuzzy Logic deals with?", "opts": ["0 and 1", "Partial truth", "Binary logic", "Predicate logic"], "a": 2, "c": "algos", "e": "Fuzzy logic allows values between 0 and 1 (partial truth).\nBinary logic is strictly 0 (false) or 1 (true)."},
            {"q": "Who proposed the Three Laws of Robotics?", "opts": ["Isaac Asimov", "Alan Turing", "Tesla", "Hinton"], "a": 1, "c": "history", "e": "Isaac Asimov coined the Three Laws in his fiction.\nThey are a foundational concept in robot ethics."},
            {"q": "Weak AI is?", "opts": ["Conscious", "Task-specific", "Failing often", "Human-level"], "a": 2, "c": "types", "e": "Weak (Narrow) AI is designed for one specific task.\nIt does not possess general consciousness or human-level reasoning."}
        ],
        "feynman": {
            "history": "FEYNMAN: Think of AI history like a family tree. John McCarthy is the 'dad' who named it. The Turing Test is just a game to see if a computer can chat so well you think it's a person. Deep Blue was a smart chess computer, and Asimov accepted the rules for robots.",
            "types": "FEYNMAN: Imagine AI tools. A screwdriver is good at one thing—that's 'Narrow' or 'Weak' AI (like Siri). A magic wand that does everything would be 'General' AI, but we don't have that yet.",
            "domains": "FEYNMAN: AI has different jobs. NLP is like the AI's ears and mouth for talking. Expert Systems are like a digital book that knows all the rules of a specific job, like a doctor or mechanic.",
            "algos": "FEYNMAN: Algorithms are recipes. A* search is a recipe to find the fastest route (like GPS). Fuzzy logic allows answers like 'sort of' instead of just yes/no."
        },
        "reassessment": {
            "history": [
                {"q": "The Dartmouth Conference (1956) is famous for?", "opts": ["Birth of AI", "Invention of Internet", "First PC", "Moon Landing"], "a": 1},
                {"q": "The Turing Test involves a judge and?", "opts": ["Two computers", "One human, one machine", "Two humans", "A robot"], "a": 2}
            ],
            "types": [
                {"q": "Siri and Alexa are examples of?", "opts": ["Strong AI", "Narrow AI", "Super AI", "General AI"], "a": 2},
                {"q": "Artificial General Intelligence (AGI) can?", "opts": ["Play chess only", "Perform any intellectual task", "Only drive cars", "Only translate"], "a": 2}
            ],
            "domains": [
                {"q": "Analyzing sentiment in text is part of?", "opts": ["Robotics", "NLP", "Vision", "Control"], "a": 2},
                {"q": "MYCIN was an early example of?", "opts": ["Game AI", "Expert System", "Chatbot", "Search Engine"], "a": 2}
            ],
            "algos": [
                {"q": "Which logic allows values between 0 and 1?", "opts": ["Boolean", "Crisp", "Fuzzy", "Binary"], "a": 3},
                {"q": "Heuristics in A* help to?", "opts": ["Slow down search", "Estimate cost to goal", "Confuse the agent", "Delete data"], "a": 2}
            ]
        }
    },
    2: { # ML
        "context": """
Machine Learning (ML) is a subset of AI that enables systems to learn from data without explicit programming.
It focuses on developing algorithms that can identify patterns and make decisions.
Key categories: Supervised Learning (labeled data), Unsupervised Learning (unlabeled data), and Reinforcement Learning (reward-based).
        """,
        "questions": [
            {"q": "Supervised Learning uses?", "opts": ["Unlabeled data", "Labeled data", "No data", "Random data"], "a": 2, "c": "basics", "e": "Supervised learning relies on input-output pairs (labels).\nUnsupervised learning deals with unlabeled data."},
            {"q": "Overfitting is when a model?", "opts": ["Learns noise/details too well", "Is too simple", "Is too slow", "Doesn't learn"], "a": 1, "c": "errors", "e": "Overfitting means the model memorized the training data.\nIt fails to generalize to new, unseen data."},
            {"q": "Logistic Regression is used for?", "opts": ["Regression", "Classification", "Clustering", "Sorting"], "a": 2, "c": "algos", "e": "Despite the name, it measures the probability of a binary outcome.\nIt classifies data into valid categories (e.g., Yes/No)."},
            {"q": "Regression predicts?", "opts": ["Classes", "Continuous values", "Clusters", "Groups"], "a": 2, "c": "basics", "e": "Regression outputs a continuous number (e.g., price, temperature).\nClassification outputs discrete class labels."},
            {"q": "K-Means is?", "opts": ["Supervised", "Unsupervised", "Reinforcement", "Semi-supervised"], "a": 2, "c": "algos", "e": "K-Means groups data without any prior labels.\nThis makes it a classic unsupervised learning algorithm."},
            {"q": "MSE is a metric for?", "opts": ["Classification", "Clustering", "Regression", "Reinforcement"], "a": 3, "c": "metrics", "e": "Mean Squared Error measures the average squared difference.\nIt applies to continuous numerical predictions (Regression)."},
            {"q": "Bias-Variance Tradeoff minimizes?", "opts": ["Speed", "Memory", "Error", "Data"], "a": 3, "c": "errors", "e": "It seeks to minimize total error by balancing bias and variance.\nExchanging one type of error for another to find the sweet spot."},
            {"q": "Regularization helps reduce?", "opts": ["Underfitting", "Overfitting", "Training time", "Data size"], "a": 2, "c": "errors", "e": "Regularization adds a penalty for complexity.\nThis discourages the model from fitting noise, thus reducing overfitting."},
            {"q": "K-Fold Cross-Validation?", "opts": ["Splits data K times", "Duplicates data", "Uses K models", "Stops training"], "a": 1, "c": "validation", "e": "It partitions data into K subsets to test stability.\nThis ensures the model handles different data splits well."},
            {"q": "KNN is a?", "opts": ["Fast learner", "Lazy learner", "Deep learner", "Recurrent learner"], "a": 2, "c": "algos", "e": "KNN does not train a model beforehand (Lazy Learner).\nIt waits until a query is made to process the data."}
        ],
        "feynman": {
            "basics": "FEYNMAN: Creating ML is like teaching a kid. Supervised learning is using flashcards with answers (labels). Regression is guessing a number (like price), Classification is picking a category (cat vs dog).",
            "errors": "FEYNMAN: Imagine memorizing a textbook word-for-word but failing a test because the questions are slightly different. That's Overfitting. Bias-Variance is balancing being too simple (rigid) vs too complex (scatterbrained).",
            "algos": "FEYNMAN: Tools in the toolbox. Logistic Regression is a switch (Yes/No). K-Means is sorting laundry into piles without knowing whose is whose (clustering). KNN is just copying your neighbors.",
            "metrics": "FEYNMAN: Keeping score. MSE calculates how far off your number guess was. Accuracy counts how many flashcards you got right.",
            "validation": "FEYNMAN: K-Fold is like taking a practice test 5 times with different questions to make sure you really know the subject, not just one specific chapter."
        },
        # Simplified reassessment assuming similar structure...
        "reassessment": {
            "basics": [{"q": "Predicting house price is?", "opts": ["Classification", "Regression", "Clustering", "Reinforcement"], "a": 2}],
            "errors": [{"q": "High Variance usually leads to?", "opts": ["Underfitting", "Overfitting", "Perfect model", "High Bias"], "a": 2}],
            "algos": [{"q": "Grouping customers by purchasing behavior (no labels) is?", "opts": ["Regression", "Clustering", "Classification", "Prediction"], "a": 2}],
            "metrics": [{"q": "Which metric is best for binary classification?", "opts": ["MSE", "Accuracy", "R-Squared", "Mean Absolute Error"], "a": 2}],
            "validation": [{"q": "Why use a validation set?", "opts": ["To train weights", "To tune hyperparameters", "To test final model", "To data mine"], "a": 2}]
        }
    },
    3: { # DL
         "context": "Deep Learning uses neural networks with many layers to learn representation from data.",
         "questions": [
             {"q": "CNNs are best for?", "opts": ["Text", "Images", "Audio", "Tables"], "a": 2, "c": "arch", "e": "CNNs use filters to scan image patterns (edges, shapes).\nThis makes them ideal for spatial data like photos."},
             {"q": "ReLU helps with?", "opts": ["Vanishing Gradient", "Overfitting", "Underfitting", "Bias"], "a": 1, "c": "foundations", "e": "ReLU does not saturate for positive values.\nThis prevents gradients from becoming too small during backprop."},
             {"q": "Optimizers do what?", "opts": ["Calc Loss", "Update Weights", "Init Weights", "Augment Data"], "a": 2, "c": "training", "e": "Optimizers adjust weights based on gradients.\nTheir goal is to minimize the loss function."},
             {"q": "RNNs handle?", "opts": ["Static data", "Sequential data", "Images", "Sparse data"], "a": 2, "c": "arch", "e": "RNNs have internal memory to remember past inputs.\nThis is crucial for sequences like text or time-series."},
             {"q": "Backpropagation?", "opts": ["Forward pass", "Calculates gradients", "Initializes", "Tests"], "a": 2, "c": "training", "e": "Backprop calculates the gradient of the loss function.\nIt allows error information to flow backward to update weights."},
             {"q": "Epoch is?", "opts": ["One batch", "One sample", "Full dataset pass", "One layer"], "a": 3, "c": "training", "e": "An epoch is one complete cycle through the training data.\nThe model sees every example once per epoch."},
             {"q": "Dropout prevents?", "opts": ["Underfitting", "Overfitting", "Slow training", "Exploding gradients"], "a": 2, "c": "training", "e": "Dropout randomly turns off neurons during training.\nThis forces the network to learn robust features, reducing overfitting."},
             {"q": "Transfer Learning?", "opts": ["Moves data", "Uses pre-trained models", "Copies code", "Is slow"], "a": 2, "c": "advanced", "e": "Transfer Learning reuses a model trained on a large dataset.\nIt saves time and works well with limited new data."},
             {"q": "GAN means?", "opts": ["Generative Adv. Network", "General AI Node", "Global Area Net", "Gradient Augment"], "a": 1, "c": "advanced", "e": "GAN stands for Generative Adversarial Network.\nIt pits a generator against a discriminator to create data."},
             {"q": "Binary Crossentropy is for?", "opts": ["Regression", "Binary Classification", "Clustering", "Multi-class"], "a": 2, "c": "foundations", "e": "Crossentropy measures error for categorical outcomes.\nBinary Crossentropy specifically handles two classes (0 or 1)."}
         ],
         "feynman": {
             "arch": "FEYNMAN: Architectures are shapes of brains. CNNs have 'eyes' for pictures. RNNs have 'memory' for stories or time.",
             "foundations": "FEYNMAN: The bricks of the brain. ReLU is a switch that says 'if negative, turn off'. Loss functions tell you how wrong you are.",
             "training": "FEYNMAN: Training is practice. Backprop is looking at your mistake and adjusting. An Epoch is reading the whole book once. Dropout is learning with one eye closed so you don't rely too much on it.",
             "advanced": "FEYNMAN: Fancy moves. Transfer Learning is using what you learned about cars to recognize trucks. GANs are two AIs playing cat and mouse to create fake images."
         },
         "reassessment": {
             "arch": [{"q": "LSTMs are an improvement on?", "opts": ["CNNs", "RNNs", "MLPs", "GANs"], "a": 2}],
             "foundations": [{"q": "Sigmoid output range?", "opts": ["0 to 1", "-1 to 1", "0 to inf", "-inf to inf"], "a": 1}],
             "training": [{"q": "Learning Rate controls?", "opts": ["Step size of update", "Number of layers", "Batch size", "Epochs"], "a": 1}],
             "advanced": [{"q": "In GANs, the Generator?", "opts": ["Classifies real data", "Creates fake data", "Calculates loss", "Optimizes weights"], "a": 2}]
         }
    },
    4: { # DS
        "context": "Data Science involves extracting insights from data using scientific methods, algorithms, and systems.",
        "questions": [
            {"q": "First step in pipeline?", "opts": ["Clean", "Collect", "Model", "Deploy"], "a": 2, "c": "process", "e": "You cannot clean or model data you don't have.\nCollection is always the necessary first step."},
            {"q": "Pandas is for?", "opts": ["Plotting", "Manipulation", "DL", "Web"], "a": 2, "c": "tools", "e": "Pandas provides DataFrames for handling structured data.\nIt is the standard tool for manipulation and analysis."},
            {"q": "EDA stands for?", "opts": ["Exploratory Data Analysis", "Error Data Admin", "Extract Data All", "End Data Access"], "a": 1, "c": "process", "e": "EDA involves summarizing main characteristics of data.\nIt often uses visualization before modeling."},
            {"q": "Histogram shows?", "opts": ["Distribution", "Relationships", "Hierarchy", "Trends"], "a": 1, "c": "viz", "e": "Histograms group data into bins to show frequency.\nThey visualize how data is distributed across a range."},
            {"q": "A Feature is?", "opts": ["Row", "Column/Variable", "Error", "Target"], "a": 2, "c": "terms", "e": "In a dataset table, rows are samples, columns are features.\nA feature represents an individual property."},
            {"q": "Missing data handling?", "opts": ["Impute", "Delete", "Ignore", "All above"], "a": 4, "c": "process", "e": "Strategies depend on the situation.\nDeletion, imputation, or ignoring are all valid options."},
            {"q": "Outlier is?", "opts": ["Normal point", "Abnormal point", "Average", "Median"], "a": 2, "c": "terms", "e": "An outlier is a data point far from others.\nIt can be caused by variability or experimental error."},
            {"q": "Correlation vs Causation?", "opts": ["Same", "Corr implies Causation", "Corr != Causation", "Causation is weak"], "a": 3, "c": "stats", "e": "Correlation only shows a statistical relationship.\nIt does not prove that one variable causes the other."},
            {"q": "SELECT is used to?", "opts": ["Delete", "Update", "Retrieve", "Insert"], "a": 3, "c": "tools", "e": "In SQL, SELECT is the command to query data.\nIt retrieves specific records from the database."},
            {"q": "3 Vs of Big Data?", "opts": ["Vol, Vel, Var", "Vis, Val, Ver", "Vec, Vor, Var", "Vol, Vis, Vec"], "a": 1, "c": "terms", "e": "Volume (size), Velocity (speed), Variety (types).\nThese define the challenges of Big Data."}
        ],
        "feynman": {
            "process": "FEYNMAN: The pipeline is a recipe. First buy ingredients (Collect), then wash them (Clean/EDA), then cook.",
            "tools": "FEYNMAN: Chef's knives. Pandas chops data. SQL finds data in the pantry.",
            "viz": "FEYNMAN: Pictures. Histograms show piles of stuff (how many apples vs oranges).",
            "terms": "FEYNMAN: Vocab. Feature = Ingredient. Outlier = Rotten apple. Big Data = Too much food for one fridge.",
            "stats": "FEYNMAN: Math sense. Correlation means things happen together (rain & umbrellas), but one didn't cause the other (umbrellas don't cause rain)."
        },
        "reassessment": {
            "process": [{"q": "Data Cleaning includes?", "opts": ["Plotting", "Fixing errors/missing values", "Training", "Deploying"], "a": 2}],
            "tools": [{"q": "Matplotlib is for?", "opts": ["Database", "Visualization", "Cleaning", "Server"], "a": 2}],
            "viz": [{"q": "Scatter plots show?", "opts": ["Frequency", "Relation between 2 vars", "Hierarchy", "Totals"], "a": 2}],
            "terms": [{"q": "The target variable is?", "opts": ["What we predict", "Input data", "Noise", "Outlier"], "a": 1}],
            "stats": [{"q": "Mean is?", "opts": ["Middle value", "Most frequent", "Average", "Range"], "a": 3}]
        }
    },
    5: { # CV
        "context": "Computer Vision trains computers to interpret and understand the visual world using digital images.",
        "questions": [
            {"q": "Basic unit of image?", "opts": ["Pixel", "Voxel", "Bit", "Byte"], "a": 1, "c": "basics", "e": "Pixel (Picture Element) is the smallest dot in an image.\nVoxels are 3D, and bits/bytes are data storage units."},
            {"q": "Screen color model?", "opts": ["CMYK", "RGB", "HSV", "BW"], "a": 2, "c": "basics", "e": "Screens use Red, Green, Blue (RGB) light to mix colors.\nPrinters typically use CMYK."},
            {"q": "YOLO stands for?", "opts": ["You Only Look Once", "You Only Live Once", "Yellow Owl", "Yield Output"], "a": 1, "c": "tasks", "e": "YOLO processes the entire image in one pass.\nThis makes it much faster than older sliding-window methods."},
            {"q": "Data Augmentation?", "opts": ["Deletes data", "Rotates/Flips", "Cleans data", "Compacts data"], "a": 2, "c": "techniques", "e": "Augmentation creates new training examples by modifying existing ones.\nRotations, flips, and zooms help the model generalize."},
            {"q": "Segmentation is?", "opts": ["Classifying whole", "Partitioning pixels", "Blurring", "Cropping"], "a": 2, "c": "tasks", "e": "Segmentation assigns a label to every pixel.\nIt separates objects from the background precisely."},
            {"q": "Canny algorithm?", "opts": ["Edge detection", "Coloring", "Sorting", "3D"], "a": 1, "c": "techniques", "e": "Canny is a multi-stage edge detection operator.\nIt finds intensity gradients to outline shapes."},
            {"q": "OCR is?", "opts": ["Text recognition", "Face recognition", " object recognition", "Edge detection"], "a": 1, "c": "tasks", "e": "Optical Character Recognition converts images of text into string data.\nIt reads license plates, documents, and signs."},
            {"q": "Face recognition feature?", "opts": ["Landmarks", "Color", "Background", "Noise"], "a": 1, "c": "techniques", "e": "Landmarks map key points (eyes, nose, mouth).\nMeasurement of distances between these points identifies a face."},
            {"q": "FPS means?", "opts": ["Frames Per Second", "First Person", "Fast Processing", "Face Scale"], "a": 1, "c": "basics", "e": "FPS measures video playback speed.\nHigher FPS results in smoother motion."},
            {"q": "Standard CV library?", "opts": ["Pandas", "OpenCV", "React", "Spark"], "a": 2, "c": "tools", "e": "OpenCV (Open Source Computer Vision) is the industry standard.\nIt contains thousands of optimized algorithms."}
        ],
        "feynman": {
            "basics": "FEYNMAN: Basics. A pixel is a dot. RGB is how screens mix Red Green Blue light to make colors. FPS is how many pictures flash per second (like a flipbook).",
            "tasks": "FEYNMAN: Jobs. YOLO finds objects fast. Segmentation traces lines around objects (like coloring book). OCR reads words.",
            "techniques": "FEYNMAN: Tricks. Augmentation is flipping photos so the AI learns 'upside down cat is still cat'. Canny finds outlines. Landmarks connect the dots on your face.",
            "tools": "FEYNMAN: OpenCV is the Swiss Army Knife for images."
        },
        "reassessment": {
            "basics": [{"q": "Grayscale images have how many channels?", "opts": ["1", "3", "4", "0"], "a": 1}],
            "tasks": [{"q": "Object Detection outputs?", "opts": ["Class only", "Bounding Box + Class", "Pixel mask", "Text"], "a": 2}],
            "techniques": [{"q": "Which removes noise?", "opts": ["Blurring", "Sharpening", "Edge detection", "Cropping"], "a": 1}],
            "tools": [{"q": "To resize an image, we use?", "opts": ["Interpolation", "Regression", "Classification", "Segmentation"], "a": 1}]
        }
    }
}

def get_input_range(prompt, set_range):
    while True:
        try:
            val = int(input(prompt))
            if val in set_range:
                return val
            print("Invalid input.")
        except ValueError:
            print("Invalid input.")

def main_loop():
    print("Autonomous Learning System")
    print("--------------------------")
    print("\nCheckpoint 1: Topic Selection")
    print("Select your learning topic:")
    for k, v in TOPICS.items():
        print(f"{k}. {v}")
    
    choice = get_input_range("Enter your choice (1-5): ", range(1, 6))
    data = CONTENT[choice]
    topic_name = TOPICS[choice]

    # Teaching Phase
    print(f"\n[{topic_name} selected]")
    print(data["context"])
    print("\nCheckpoint Quiz Started")

    # Initial Quiz
    score = 0
    wrong_concepts = []
    user_results = [] # Store tuple of (Question Index, User Answer, Correct Answer, Explanation, IsCorrect)

    for i, q in enumerate(data["questions"], 1):
        print(f"\nQuestion {i}: {q['q']}")
        for j, opt in enumerate(q["opts"], 1):
            print(f"{j}. {opt}")
        
        ans = get_input_range("Enter your answer (1-4): ", range(1, 5))
        is_correct = (ans == q["a"])
        
        user_results.append({
            "id": i,
            "user_ans": ans,
            "correct_ans": q["a"],
            "correct_status": "CORRECT" if is_correct else "WRONG",
            "explanation": q.get("e", "Review this topic safely."), # Fallback if key missing
            "is_correct": is_correct
        })

        if is_correct:
            score += 1
        else:
            wrong_concepts.append(q["c"])
            
    # --- QUIZ DIAGNOSTIC REPORT ---
    print("\n====================================")
    print("QUIZ DIAGNOSTIC REPORT")
    print("====================================")
    
    for res in user_results:
        print(f"Q{res['id']}: User Answer = {res['user_ans']} | Correct Answer = {res['correct_ans']} | {res['correct_status']}")
        if not res['is_correct']:
            print("Explanation:")
            print(res['explanation'])
            print() # Extra newline for spacing between items
        else:
            # If correct, we add a clear logical break (or just newline)
            pass

    percentage = (score / 10) * 100

    # Decision Logic
    if percentage >= 70:
        print("\n------------------------------")
        print("QUIZ END")
        print("------------------------------")
        print(f"Final Score: {score} / 10")
        print("Checkpoint Status: PASSED")
        print("Message: You have demonstrated sufficient understanding.")
    else:
        print("\n------------------------------")
        print("CHECKPOINT FAILED")
        print("------------------------------")
        print(f"Final Score: {score} / 10")
        print("Required Score: 7 / 10")
        print("Status: Relearning Required")
        print("Initiating Feynman Technique...")
        
        # Feynman Phase
        print("\n================================")
        print("FEYNMAN RE-EXPLANATION")
        print("================================")
        
        unique_failed_concepts = list(set(wrong_concepts))
        
        for c in unique_failed_concepts:
            key = c
            explanation = data["feynman"].get(key, "Review this concept simply.")
            print(f"\n[Concept: {key.upper()}]")
            print(explanation)
        
        print("\nReassessment Started")
        
        # Reassessment Phase
        # Generate 5 questions from weak concepts
        re_questions = []
        
        # Strategy: Iterate through weak concepts and pick questions. 
        # Repeat cycle until we have 5.
        
        pool = []
        for c in unique_failed_concepts:
            if c in data["reassessment"]:
                # Inject 'c' into these questions so we can track failures
                for q_obj in data["reassessment"][c]:
                    q_with_c = q_obj.copy()
                    q_with_c['c'] = c
                    pool.append(q_with_c)
        
        # If pool is empty (shouldn't be), fallback to random initial questions?
        if not pool:
             pool = data["questions"] # Fallback

        # Select 5
        # To ensure we target weak areas, we prioritize the pool.
        # If pool < 5, we might repeat or fill.
        
        if len(pool) >= 5:
            re_questions = random.sample(pool, 5)
        else:
            # If not enough, take all and fill with randoms from initial (excluding correctly answered? no, simplified)
            re_questions = pool
            # duplicate to fill?
            while len(re_questions) < 5:
                re_questions.append(random.choice(pool))

        re_score = 0
        re_wrong_concepts = []

        for i, q in enumerate(re_questions, 1):
            print(f"\nRe-Question {i}: {q['q']}")
            for j, opt in enumerate(q["opts"], 1):
                print(f"{j}. {opt}")
            
            ans = get_input_range("Enter your answer (1-4): ", range(1, 5))
            if ans == q["a"]:
                re_score += 1
            else:
                # Safely get 'c' if it exists
                if 'c' in q:
                    re_wrong_concepts.append(q['c'])
        
        re_percentage = (re_score / 5) * 100
        
        if re_percentage >= 70:
            print("\n------------------------------")
            print("CHECKPOINT PASSED AFTER RELEARNING")
            print("------------------------------")
            print(f"Final Score: {re_score} / 5")
            print("Message: Concepts are now clear.")
        else:
            print("\n------------------------------")
            print("CHECKPOINT STILL NOT CLEARED")
            print("------------------------------")
            
            # Final Feynman Review
            if re_wrong_concepts:
                print("Final Feynman Review for missed concepts:")
                unique_re_failed = list(set(re_wrong_concepts))
                for c in unique_re_failed:
                    explanation = data["feynman"].get(c, "Review this concept.")
                    # Strip the "FEYNMAN: " prefix if present for a cleaner "Remember" flow, or keep it.
                    # We will just print it directly.
                    print(f"\n[Concept: {c.upper()}]")
                    print(f"Remember: {explanation}")
            
            print("\nRecommendation: Restart learning module.")

if __name__ == "__main__":
    main_loop()
