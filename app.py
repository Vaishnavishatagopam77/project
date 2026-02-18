import streamlit as st
import time
import random

# ==========================================
# CONTENT REPOSITORY (DETAILED MODULES)
# ==========================================

TOPICS = {
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    "DS": "Data Science",
    "CV": "Computer Vision"
}

CONTENT = {
    "AI": {
        "context": """
### 1. Module Overview
Artificial Intelligence (AI) is the broad science of mimicking human abilities. It is not just about robots; it is about creating systems that can reason, learn, and act autonomously to solve complex problems. AI exists to automate intellectual tasks, ranging from planning a logistics route to diagnosing diseases.

### 2. Core Concepts
**Intelligent Agents**: An agent is any entity that perceives its environment through sensors and acts upon that environment through actuators to achieve a goal. A thermostat is a simple agent; a self-driving car is a complex one.
**Search Strategies**: Before an AI can act, it often needs to search for the best sequence of actions. Algorithms like Breadth-First Search (BFS) or A* (A-Star) simulate potential futures to find the optimal path to a goal.
**Knowledge Representation**: For an AI to reason, it needs a way to store what it knows. This involves logic (like 'If it rains, the grass is wet') and structured data (ontologies) that allow the machine to deduce new information.
**Reasoning and Planning**: Once knowledge is stored, the AI uses logic to infer new facts and planning algorithms to construct a sequence of actions that transition the world from a starting state to a goal state.

### 3. Workflow / Process
1. **Perception**: The system gathers raw data (text, sensor readings, images).
2. **processing**: Data is converted into a structured format the internal model understands.
3. **Reasoning/Decision**: The agent uses its knowledge base or models to select the best action.
4. **Action**: The decision is executed/actuated.
5. **Feedback Loop**: The agent observes the result of the action and updates its state.

### 4. Tools & Techniques
- **Heuristic Search**: Using 'rules of thumb' to find solutions faster than verifying every possibility.
- **Logic Programming**: Using languages like Prolog where relationships are defined, and the computer solves for the answer.
- **Minimax Algorithm**: A decision rule used in game theory (like Chess) to minimize the possible loss for a worst case scenario.

### 5. Real-World Applications
- **Virtual Assistants**: Siri and Alexa process speech (perception), understand intent (reasoning), and execute commands (action).
- **Game Playing**: Systems like DeepBlue (Chess) and AlphaGo use advanced search and planning to defeat humans.
- **Logistics**: FedEx and Amazon use AI planning to optimize delivery routes dynamically.
        """,
        "questions": [
            {"id": 1, "q": "What is an Intelligent Agent?", "opts": ["A secret spy", "Entity that perceives and acts", "A database", "A robot arm"], "a": "Entity that perceives and acts", "c": "concepts", "e": "Agents are defined by perception and action cycles."},
            {"id": 2, "q": "A* Search uses heuristics to...", "opts": ["Slow down", "Find shortest path efficiently", "Randomize path", "Encrypt data"], "a": "Find shortest path efficiently", "c": "tools", "e": "Heuristics guide the search to the goal closer, saving time."},
            {"id": 3, "q": "Knowledge Representation involves?", "opts": ["Storing logic/facts", "Storing pixels", "Storing electricity", "Representing colors"], "a": "Storing logic/facts", "c": "concepts", "e": "It structure information so the AI can reason about it."},
            {"id": 4, "q": "The Minimax algorithm is used in?", "opts": ["Sorting", "Adversarial Games", "Painting", "Data cleaning"], "a": "Adversarial Games", "c": "tools", "e": "It minimizes the maximum possible loss, critical for games like Chess."},
            {"id": 5, "q": "Perception in AI refers to?", "opts": ["Feeling emotions", "Gathering raw input", "Moving motors", "Printing logs"], "a": "Gathering raw input", "c": "workflow", "e": "Perception is the input stage (sensors, cameras, microphones)."},
            {"id": 6, "q": "Who coined the term AI?", "opts": ["Turing", "McCarthy", "Musk", "Altman"], "a": "McCarthy", "c": "history", "e": "John McCarthy at the Dartmouth Conference in 1956."},
            {"id": 7, "q": "What is a Heuristic?", "opts": ["A rule of thumb", "A hardware chip", "A coding error", "A database key"], "a": "A rule of thumb", "c": "tools", "e": "Heuristics allow faster decisions by trading optimality for speed."},
            {"id": 8, "q": "Logic Programming uses?", "opts": ["Imperative steps", "Facts and Rules", "Pixels", "Audio waves"], "a": "Facts and Rules", "c": "tools", "e": "It defines 'what' is true, not 'how' to calculate it."},
            {"id": 9, "q": "Planning in AI means?", "opts": ["Scheduling meetings", "Sequence of actions to goal", "Drawing blueprints", "Predicting stock"], "a": "Sequence of actions to goal", "c": "concepts", "e": "Planning constructs a path of states from start to finish."},
            {"id": 10, "q": "Strong AI refers to?", "opts": ["High computational power", "Human-level consciousness", "Durable hardware", "Good at Chess"], "a": "Human-level consciousness", "c": "concepts", "e": "Strong AI (AGI) can apply intelligence to any problem, like a human."}
        ],
        "feynman": {
            "concepts": "FEYNMAN: Think of an agent like a butler. It sees the messy room (perception), knows it should be clean (knowledge), decides to vacuum (reasoning), and does it (action).",
            "tools": "FEYNMAN: Heuristics are like educated guesses. Instead of checking every single street to find your house, you check the neighborhood that looks right first.",
            "workflow": "FEYNMAN: Sense -> Think -> Act. Eyes see the ball, Brain calculates the catch, Hand moves to grab it.",
            "history": "FEYNMAN: McCarthy is the dad of AI. He named the baby in 1956."
        },
        "reassessment": {
            "concepts": [{"q": "An agent must have?", "opts": ["Sensors and Actuators", "Legs", "Internet", "Screen"], "a": "Sensors and Actuators"}],
            "tools": [{"q": "In A* search, 'h(n)' stands for?", "opts": ["Heuristic", "History", "Hardware", "Height"], "a": "Heuristic"}],
            "workflow": [{"q": "First step of AI workflow?", "opts": ["Action", "Perception", "Reasoning", "Sleep"], "a": "Perception"}]
        }
    },
    "ML": {
        "context": """
### 1. Module Overview
Machine Learning (ML) is the field where computers learn without being explicitly programmed. Instead of writing rules ("If email contains 'buy now', mark spam"), we feed the system thousands of emails and let it figure out the rules itself. It solves problems where the rules are too complex for humans to write down manually.

### 2. Core Concepts
**Learning from Data**: ML algorithms parse data, learn from it, and make determinations or predictions about something in the world.
**Types of Learning**:
- *Supervised*: Learning with a teacher. The data has labels (Input: Image, Label: "Cat").
- *Unsupervised*: Learning self-discovery. The data has no labels; the model finds structure (Cluster: "These customers buy similar things").
- *Reinforcement*: Learning by trial and error. An agent gets rewards for good actions and penalties for bad ones.
**Overfitting**: When a model memorizes the training data like a student memorizing textbook answers. It fails on the final exam (new data) because it didn't learn the concepts, just the specific examples.

### 3. Workflow / Process
1. **Data Collection**: Gathering raw data.
2. **Preprocessing**: Cleaning data, filling missing values, normalizing numbers.
3. **Training**: feeding data into the algorithm to adjust internal weights.
4. **Evaluation**: Testing the model on data it hasn't seen to check accuracy.
5. **Deployment**: Using the model in the real world to make predictions.

### 4. Tools & Techniques
- **Regression**: Predicting a continuous number (e.g., House Price).
- **Classification**: Predicting a category (e.g., Spam vs Not Spam).
- **Clustering**: Grouping similar items (e.g., Customer Segmentation).
- **Evaluation Metrics**: Accuracy (correctness), Precision/Recall (specific types of correctness for messy data).

### 5. Real-World Applications
- **Recommendation Engines**: Netflix and YouTube analyze your history to suggest new content.
- **Fraud Detection**: Banks use ML to flag transactions that deviate from your normal spending patterns.
- **Medical Diagnosis**: Predicting disease risk based on patient history and test results.
        """,
        "questions": [
            {"id": 1, "q": "Supervised Learning requires?", "opts": ["Labeled Data", "Unlabeled Data", "No Data", "Rewards"], "a": "Labeled Data", "c": "concepts", "e": "Supervised means inputs come with the correct answers (labels)."},
            {"id": 2, "q": "Predicting House Price is?", "opts": ["Classification", "Regression", "Clustering", "Reinforcement"], "a": "Regression", "c": "tools", "e": "Prices are continuous numbers, so it is Regression."},
            {"id": 3, "q": "Overfitting happens when?", "opts": ["Model is too simple", "Model memorizes training data", "Data is missing", "Training is fast"], "a": "Model memorizes training data", "c": "concepts", "e": "The model learns noise instead of the signal, failing to generalize."},
            {"id": 4, "q": "Grouping customers without labels is?", "opts": ["Supervised", "Unsupervised", "Reinforcement", "Regression"], "a": "Unsupervised", "c": "concepts", "e": "Unsupervised learning finds hidden patterns/clusters without labels."},
            {"id": 5, "q": "Step after Training is?", "opts": ["Collection", "Evaluation", "Cleaning", "Deployment"], "a": "Evaluation", "c": "workflow", "e": "You must test/evaluate the model to ensure it works before deployment."},
            {"id": 6, "q": "Accuracy measures?", "opts": ["Speed", "Percentage of correct predictions", "Memory usage", "Dataset size"], "a": "Percentage of correct predictions", "c": "tools", "e": "Accuracy = Correct Guesses / Total Guesses."},
            {"id": 7, "q": "Reinforcement Learning uses?", "opts": ["Labels", "Rewards/Penalties", "Clusters", "SQL"], "a": "Rewards/Penalties", "c": "concepts", "e": "The agent learns by maximizing cumulative reward."},
            {"id": 8, "q": "Spam filtering is what task?", "opts": ["Clustering", "Regression", "Classification", "Planning"], "a": "Classification", "c": "tools", "e": "It classifies emails into two categories: Spam or Ham."},
            {"id": 9, "q": "Data Preprocessing involves?", "opts": ["Buying computers", "Cleaning and Normalizing", "Selling data", "Printing"], "a": "Cleaning and Normalizing", "c": "workflow", "e": "Raw data is often messy and needs preparation before training."},
            {"id": 10, "q": "A model that fails on new data likely has?", "opts": ["Underfitting", "Overfitting", "Good luck", "No data"], "a": "Overfitting", "c": "concepts", "e": "Poor generalization is the hallmark of overfitting."}
        ],
        "feynman": {
            "concepts": "FEYNMAN: Supervised is like flashcards with answers on the back. Unsupervised is like sorting lego blocks by color without anyone telling you the color names. Overfitting is memorizing the practice test but failing the real exam.",
            "workflow": "FEYNMAN: Gather ingredients (Data), Chop them (Preprocess), Cook (Train), Taste test (Evaluate), Serve (Deploy).",
            "tools": "FEYNMAN: Regression guesses a number (Height). Classification guesses a name (Boy/Girl)."
        },
        "reassessment": {
            "concepts": [{"q": "If data has no target label, use?", "opts": ["Supervised", "Unsupervised", "Regression", "Checking"], "a": "Unsupervised"}],
            "workflow": [{"q": "Cleaning missing values happens in?", "opts": ["Deployment", "Preprocessing", "Evaluation", "None"], "a": "Preprocessing"}],
            "tools": [{"q": "Predicting if it will rain (Yes/No)?", "opts": ["Regression", "Classification", "Clustering", "Summarization"], "a": "Classification"}]
        }
    },
    "DL": {
        "context": """
### 1. Module Overview
Deep Learning (DL) is a specialized subset of Machine Learning inspired by the structure of the human brain. While standard ML needs humans to define features (e.g., "look for round edges"), DL learns the features itself using multi-layered Neural Networks. It has revolutionized fields like image recognition and language translation.

### 2. Core Concepts
**Neural Networks**: Networks of artificial 'neurons' connected in layers. Each connection has a 'weight' that adjusts during training.
**Layers**:
- *Input Layer*: Receives raw data.
- *Hidden Layers*: Transform data into abstract features. Deep learning means 'many hidden layers'.
- *Output Layer*: Gives the final prediction.
**Backpropagation**: The 'learning' mechanism. The network compares its guess to the real answer, calculates the error (loss), and works backward to adjust the weights to reduce the error next time.
**Activation Functions**: Mathematical functions (like ReLU or Sigmoid) that decide if a neuron 'fires', adding non-linearity to the model.

### 3. Workflow / Process
1. **Architecture Design**: Choosing the number of layers and neurons.
2. **Forward Propagation**: Passing data through the network to get a guess.
3. **Loss Calculation**: Measuring how wrong the guess was.
4. **Backpropagation**: Updating the weights to fix the error.
5. **Iterate**: Repeating millions of times (Epochs) until accurate.

### 4. Tools & Techniques
- **CNN (Convolutional Neural Networks)**: Specialized for grids like images. They scan for edges, textures, and shapes.
- **RNN (Recurrent Neural Networks)**: Specialized for sequences like text or time-series. They have 'memory' of previous inputs.
- **Transformers**: The modern architecture behind GPT and BERT, using 'attention' mechanisms to process entire sequences in parallel.

### 5. Real-World Applications
- **Facial Recognition**: Unlocking phones by analyzing pixel patterns of faces.
- **Machine Translation**: Google Translate converting sentences while keeping context.
- **Generative AI**: Tools like Midjourney creating art from text descriptions.
        """,
        "questions": [
            {"id": 1, "q": "Deep Learning is inspired by?", "opts": ["Statistics", "Human Brain", "Physics", "Geology"], "a": "Human Brain", "c": "concepts", "e": "Neural networks mimic biological neurons and synapses."},
            {"id": 2, "q": "Backpropagation is used to?", "opts": ["Make predictions", "Update weights/learn", "Initialize network", "Delete data"], "a": "Update weights/learn", "c": "concepts", "e": "It propagates error backwards to adjust weights and minimize loss."},
            {"id": 3, "q": "CNNs are best for?", "opts": ["Text", "Images/Video", "Spreadsheets", "Audio"], "a": "Images/Video", "c": "tools", "e": "Convolutional layers are designed to capture spatial hierarchies in images."},
            {"id": 4, "q": "Hidden Layers are located?", "opts": ["Before Input", "Between Input and Output", "After Output", "In the cloud"], "a": "Between Input and Output", "c": "concepts", "e": "They perform the intermediate processing/feature extraction."},
            {"id": 5, "q": "One pass through the whole dataset is?", "opts": ["Batch", "Step", "Epoch", "Layer"], "a": "Epoch", "c": "workflow", "e": "An epoch is one complete cycle of training on the full dataset."},
            {"id": 6, "q": "RNNs differ from CNNs because they have?", "opts": ["Memory/Loops", "Filters", "Colors", "No weights"], "a": "Memory/Loops", "c": "tools", "e": "RNNs handle sequential data by retaining state from previous steps."},
            {"id": 7, "q": "ReLU is an example of?", "opts": ["Loss Function", "Optimizer", "Activation Function", "Layer"], "a": "Activation Function", "c": "concepts", "e": "Activation functions introduce non-linearity into the network."},
            {"id": 8, "q": "Loss Calculation measures?", "opts": ["Network speed", "Error/Difference", "Electricity", "Memory"], "a": "Error/Difference", "c": "workflow", "e": "Loss quantifies how far the prediction is from the actual target."},
            {"id": 9, "q": "Transformers use what mechanism?", "opts": ["Convolution", "Attention", "Regression", "Forgetting"], "a": "Attention", "c": "tools", "e": "Self-attention allows the model to weigh the importance of different words."},
            {"id": 10, "q": "Deep Learning requires?", "opts": ["Manual features", "Large data & compute", "Simple rules", "No training"], "a": "Large data & compute", "c": "overview", "e": "DL models are data-hungry and computationally expensive to train."}
        ],
        "feynman": {
            "concepts": "FEYNMAN: A Neural Net is like a team of detectives passing a file. Each layer adds a detail. Layer 1 says 'I see lines'. Layer 2 says 'The lines make an eye'. Backprop is the boss yelling 'You're wrong!' so they do better next time.",
            "workflow": "FEYNMAN: Guess (Forward), Check Score (Loss), Learn from mistakes (Backward), Repeat.",
            "tools": "FEYNMAN: CNN is for eyes (Pictures). RNN is for memory (Stories). Transformers are the speed-readers."
        },
        "reassessment": {
            "concepts": [{"q": "Weights are adjusted during?", "opts": ["Forward Prop", "Backpropagation", "Input", "Testing"], "a": "Backpropagation"}],
            "workflow": [{"q": "Forward propagation produces?", "opts": ["Prediction", "Error", "Weights", "Gradients"], "a": "Prediction"}],
            "tools": [{"q": "To process a sentence, use?", "opts": ["CNN", "RNN", "K-Means", "Linear Regression"], "a": "RNN"}]
        }
    },
    "DS": {
        "context": """
### 1. Module Overview
Data Science (DS) is the interdisciplinary field of extracting knowledge and insights from data. It combines coding, statistics, and domain expertise. While ML focuses on making predictions, Data Science focuses on *understanding* the data to drive decision-making.

### 2. Core Concepts
**Data Lifecycle**: The journey from raw data to actionable insight.
**Exploratory Data Analysis (EDA)**: The detective phase. Using stats and plots to find patterns, anomalies, and correlations before doing any modeling.
**Statistics Intuition**: Understanding distributions (Bell curve), probability, and hypothesis testing to know if a finding is real or just luck.
**Data Cleaning**: The most time-consuming part. Fixing missing values, typos, and format errors. "Garbage In, Garbage Out".

### 3. Workflow / Process (OSEMN)
1. **Obtain**: scraping, querying databases (SQL), or logging APIs.
2. **Scrub**: Cleaning data, handling missing values.
3. **Explore**: Plotting histograms, finding correlations (EDA).
4. **Model**: building predictive models (ML) or statistical models.
5. **iNterpret**: Explaining the results to stakeholders (Visualization/Storytelling).

### 4. Tools & Techniques
- **Pandas/DataFrames**: The standard tool for manipulating table-like data.
- **Visualization**: Using charts (Scatter, Bar, Heatmap) to communicate findings instantly.
- **SQL**: The language used to talk to databases and retrieve specific data chunks.

### 5. Real-World Applications
- **Business Intelligence**: Dashboards showing live sales trends to help CEOs make decisions.
- **Public Policy**: Analyzing traffic data to decide where to build new roads.
- **Sports Analytics**: Analyzing player stats to decide which athlete to draft (Moneyball).
        """,
        "questions": [
            {"id": 1, "q": "What does EDA stand for?", "opts": ["Exploratory Data Analysis", "External Data Access", "Error Detection Algorithm", "Easy Data Action"], "a": "Exploratory Data Analysis", "c": "concepts", "e": "EDA is the process of summarizing and visualizing main characteristics."},
            {"id": 2, "q": "The first step in OSEMN is?", "opts": ["Model", "Obtain", "Scrub", "Interpret"], "a": "Obtain", "c": "workflow", "e": "You must obtain data (collect/query) before you can do anything else."},
            {"id": 3, "q": "Garbage In, Garbage Out refers to?", "opts": ["Recycling", "Bad data leads to bad results", "Deleting files", "Formatting hard drives"], "a": "Bad data leads to bad results", "c": "concepts", "e": "Quality of output is determined by quality of input data."},
            {"id": 4, "q": "SQL is used for?", "opts": ["Plotting", "Querying Databases", "Web Design", "Training Neural Nets"], "a": "Querying Databases", "c": "tools", "e": "SQL (Structured Query Language) retrieves data from relational DBs."},
            {"id": 5, "q": "Data Cleaning involves?", "opts": ["Wiping the screen", "Fixing missing/error values", "Deleting all data", "Compressing files"], "a": "Fixing missing/error values", "c": "workflow", "e": "Scrubbing data ensures constraints and validity are met."},
            {"id": 6, "q": "Visualization is key for?", "opts": ["Interpretation/Communication", "Storage", "Encryption", "Networking"], "a": "Interpretation/Communication", "c": "tools", "e": "Humans understand visual patterns faster than raw tables of numbers."},
            {"id": 7, "q": "Pandas is a tool for?", "opts": ["Zoos", "Data Manipulation", "3D Printing", "Sound Editing"], "a": "Data Manipulation", "c": "tools", "e": "Pandas is the primary Python library for data analysis."},
            {"id": 8, "q": "A Heatmap is a type of?", "opts": ["Database", "Visualization", "Server", "Algorithm"], "a": "Visualization", "c": "tools", "e": "Heatmaps show magnitude of phenomena as color in two dimensions."},
            {"id": 9, "q": "In DS, Domain Expertise means?", "opts": ["Knowing the specific industry", "Owning a website", "Being an admin", "Knowing math"], "a": "Knowing the specific industry", "c": "overview", "e": "You need to understand the subject (e.g., medicine) to interpret the data."},
            {"id": 10, "q": "Which is NOT a step in OSEMN?", "opts": ["Obtain", "Scrub", "Negotiate", "Model"], "a": "Negotiate", "c": "workflow", "e": "Negotiate is not part of the standard data science lifecycle."}
        ],
        "feynman": {
            "concepts": "FEYNMAN: Data Cleaning is like washing vegetables before cooking. If you cook with dirt, the meal (Insigts) tastes bad. EDA is looking at the ingredients to see what you have before deciding on a recipe.",
            "workflow": "FEYNMAN: Get it -> CLean it -> Look at it -> Math it -> Explain it.",
            "tools": "FEYNMAN: SQL is asking the library for a specific book. Visualization is drawing a picture because a definition is too boring."
        },
        "reassessment": {
            "concepts": [{"q": "Why clean data?", "opts": ["To save space", "To ensure accuracy", "To make it pretty", "To hide it"], "a": "To ensure accuracy"}],
            "workflow": [{"q": "Last step of DS workflow?", "opts": ["Scrub", "Interpret/Act", "Obtain", "Model"], "a": "Interpret/Act"}],
            "tools": [{"q": "Best chart for correlation?", "opts": ["Pie Chart", "Scatter Plot", "Bar Chart", "List"], "a": "Scatter Plot"}]
        }
    },
    "CV": {
        "context": """
### 1. Module Overview
Computer Vision (CV) is the field of teaching computers to "see" and interpret the visual world. Just as humans use eyes and brains to understand a scene, CV uses cameras and algorithms to identify objects, track movement, and reconstruct 3D environments.

### 2. Core Concepts
**Pixels**: The atoms of an image. A digital image is a grid of numbers (0-255) representing brightness or color channels (Red, Green, Blue).
**Feature Extraction**: Detecting key parts of an image, like edges, corners, or blobs, that define an object to the computer.
**Object Detection**: Not just *what* is in the image (Classification), but *where* it is (Bounding Box).
**Segmentation**: Classifying every single pixel to trace exact outlines of objects (e.g., separating the road from the sidewalk).

### 3. Workflow / Process
1. **Acquisition**: Taking the photo or video frame.
2. **Preprocessing**: Resizing, changing to grayscale, removing noise/blur.
3. **Feature Extraction**: Finding lines, shapes, or deep features.
4. **Analysis/Recognition**: Matching features to known templates or using a model to classify.
5. **Action**: Making a decision (e.g., car hits brakes).

### 4. Tools & Techniques
- **Edge Detection (Canny)**: Math that finds where brightness changes sharply (boundaries).
- **YOLO (You Only Look Once)**: A famous fast algorithm for real-time object detection.
- **OpenCV**: The standard open-source library containing thousands of vision utilities.
- **Data Augmentation**: Flipping and rotating images to give the AI more examples to learn from.

### 5. Real-World Applications
- **Self-Driving Cars**: Using CV to see lanes, signs, and pedestrians.
- **Medical Imaging**: Detecting tumors in X-rays or MRI scans better than human eyes.
- **Retail**: Amazon Go stores tracking what items you pick off the shelf automatically.
        """,
        "questions": [
            {"id": 1, "q": "A pixel value of 0 usually means?", "opts": ["White", "Black", "Red", "Transparent"], "a": "Black", "c": "concepts", "e": "In grayscale, 0 is no intensity (black), 255 is full intensity (white)."},
            {"id": 2, "q": "Object Detection finds?", "opts": ["Class only", "Class and Location", "Pixel mask", "Resolution"], "a": "Class and Location", "c": "concepts", "e": "It draws a bounding box around objects and labels them."},
            {"id": 3, "q": "YOLO stands for?", "opts": ["You Only Live Once", "You Only Look Once", "Yellow Owl", "Yield Output"], "a": "You Only Look Once", "c": "tools", "e": "It refers to processing the image in a single pass for speed."},
            {"id": 4, "q": "Segmentation is?", "opts": ["Classifying the whole image", "Classifying pixels", "Cropping", "Zooming"], "a": "Classifying pixels", "c": "concepts", "e": "Semantic segmentation assigns a class to every pixel."},
            {"id": 5, "q": "Edge Detection looks for?", "opts": ["Colors", "Sharp brightness changes", "Smooth areas", "Faces"], "a": "Sharp brightness changes", "c": "tools", "e": "Edges are defined by discontinuities in image intensity."},
            {"id": 6, "q": "Preprocessing in CV includes?", "opts": ["Resizing/Grayscale", "Deleting camera", "Compiling code", "Networking"], "a": "Resizing/Grayscale", "c": "workflow", "e": "Standardizing input size and color space is crucial for models."},
            {"id": 7, "q": "OpenCV is?", "opts": ["A VR headset", "A CV Library", "A camera brand", "A robot"], "a": "A CV Library", "c": "tools", "e": "OpenCV provides standard algorithms for image processing."},
            {"id": 8, "q": "Data Augmentation helps?", "opts": ["Reduce dataset size", "Model Generalization", "Make images smaller", "Blur images"], "a": "Model Generalization", "c": "tools", "e": "It creates variations so the model is robust to rotation/zoom."},
            {"id": 9, "q": "An image is essentially?", "opts": ["A list of words", "A grid of numbers", "A sound wave", "A vector"], "a": "A grid of numbers", "c": "concepts", "e": "Computers see images as matrices of pixel intensity values."},
            {"id": 10, "q": "Self-driving cars use CV to?", "opts": ["Listen to radio", "Detect lanes/obstacles", "Cool the engine", "Check oil"], "a": "Detect lanes/obstacles", "c": "overview", "e": "CV is the primary sensor modality for understanding road geometry."}
        ],
        "feynman": {
            "concepts": "FEYNMAN: A digital picture is just a giant Lite-Brite grid. The computer looks at the numbers behind the lights. Detection is drawing a box; Segmentation is coloring inside the lines.",
            "workflow": "FEYNMAN: See -> Clean -> Find Lines -> Recognize -> Act.",
            "tools": "FEYNMAN: Edge detection is just connecting the dots where the color jumps. YOLO is a fast glance that spots everything at once."
        },
        "reassessment": {
            "concepts": [{"q": "RGB stands for?", "opts": ["Red Green Blue", "Real Good Byte", "Radio Graphic Base", "Red Gray Black"], "a": "Red Green Blue"}],
            "workflow": [{"q": "Removing noise happens in?", "opts": ["Acquisition", "Preprocessing", "Action", "None"], "a": "Preprocessing"}],
            "tools": [{"q": "Annotating every pixel is?", "opts": ["Detection", "Segmentation", "Classification", "Regression"], "a": "Segmentation"}]
        }
    }
}

# Fallback for other topics to prevent crash
DEFAULT_CONTENT = CONTENT["AI"]

# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================

if 'stage' not in st.session_state:
    st.session_state.stage = 'topic_selection'
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = None
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_q_index' not in st.session_state:
    st.session_state.quiz_q_index = 0
if 'weak_concepts' not in st.session_state:
    st.session_state.weak_concepts = []
if 'mastery_status' not in st.session_state:
    st.session_state.mastery_status = "PENDING"

# ==========================================
# NAVIGATION FUNCTIONS
# ==========================================

def set_topic(topic_code):
    st.session_state.selected_topic = topic_code
    st.session_state.stage = 'teaching'
    st.session_state.quiz_q_index = 0
    st.session_state.user_answers = {}
    st.session_state.score = 0
    st.session_state.mastery_status = "PENDING"

def start_quiz():
    st.session_state.stage = 'quiz'
    st.session_state.quiz_q_index = 0

def submit_answer(question_id, answer):
    st.session_state.user_answers[question_id] = answer
    # Move to next question or finish
    topic_data = CONTENT.get(st.session_state.selected_topic, DEFAULT_CONTENT)
    total_q = len(topic_data['questions'])
    
    if st.session_state.quiz_q_index < total_q - 1:
        st.session_state.quiz_q_index += 1
    else:
        st.session_state.stage = 'report'
        calculate_score()

def calculate_score():
    topic_data = CONTENT.get(st.session_state.selected_topic, DEFAULT_CONTENT)
    questions = topic_data['questions']
    
    correct_count = 0
    st.session_state.weak_concepts = []
    
    for q in questions:
        u_ans = st.session_state.user_answers.get(q['id'])
        if u_ans == q['a']:
            correct_count += 1
        else:
            if 'c' in q:
                st.session_state.weak_concepts.append(q['c'])
    
    st.session_state.score = (correct_count / len(questions)) * 100
    if st.session_state.score >= 70:
        st.session_state.mastery_status = "PASSED"
    else:
        st.session_state.mastery_status = "FAILED"

def start_feynman():
    st.session_state.stage = 'feynman'

def start_reassessment():
    st.session_state.stage = 'reassessment'
    st.session_state.user_answers = {} # Reset for new quiz

def submit_reassessment(answers):
    # Logic to evaluate reassessment
    topic_data = CONTENT.get(st.session_state.selected_topic, DEFAULT_CONTENT)
    correct = 0
    total = len(answers)
    
    # Simple check - in real app, match IDs
    # Here we assume answers passed are (Question dict, User Answer)
    for q, ans in answers:
        if ans == q['a']:
            correct += 1
            
    final_score = (correct / total) * 100 if total > 0 else 0
    
    if final_score >= 70:
        st.session_state.mastery_status = "PASSED_RELEARN"
    else:
        st.session_state.mastery_status = "FAILED_FINAL"
        
    st.session_state.score = final_score
    st.session_state.stage = 'final'


# ==========================================
# STAGE RENDERERS
# ==========================================

def render_topic_selection():
    st.title("Autonomous Learning System")
    st.markdown("### Select your learning path")
    
    topic = st.selectbox("Choose Topic", list(TOPICS.keys()), format_func=lambda x: TOPICS[x])
    
    if st.button("Start Learning"):
        set_topic(topic)
        st.rerun()

def render_teaching():
    topic_code = st.session_state.selected_topic
    full_name = TOPICS.get(topic_code, "Unknown")
    data = CONTENT.get(topic_code, DEFAULT_CONTENT)
    
    st.title(f"Module: {full_name}")
    st.markdown("---")
    st.info("Read the following carefully before proceeding.")
    
    st.markdown(data['context'])
    
    st.markdown("---")
    if st.button("Start Quiz"):
        start_quiz()
        st.rerun()

def render_quiz():
    st.title("Checkpoint Quiz")
    topic_data = CONTENT.get(st.session_state.selected_topic, DEFAULT_CONTENT)
    questions = topic_data['questions']
    
    idx = st.session_state.quiz_q_index
    q = questions[idx]
    
    # Progress
    st.progress((idx + 1) / len(questions))
    st.caption(f"Question {idx + 1} of {len(questions)}")
    
    st.markdown(f"### {q['q']}")
    
    # Uses a key based on index to ensure widget uniqueness
    choice = st.radio("Select an option:", q['opts'], key=f"q_{idx}")
    
    if st.button("Submit Answer"):
        submit_answer(q['id'], choice)
        st.rerun()

def render_report():
    st.title("Diagnostic Report")
    
    score = st.session_state.score
    status = st.session_state.mastery_status
    
    # Progress bar for score visualization
    st.progress(score / 100)
    
    if status == "PASSED":
        st.success(f"### Checkpoint Passed! Score: {score:.1f}%")
        st.balloons()
    else:
        st.error(f"### Checkpoint Failed. Score: {score:.1f}%")
        st.markdown(f"**Required: 70%** - You scored below the mastery threshold.")
    
    st.markdown("---")
    
    topic_data = CONTENT.get(st.session_state.selected_topic, DEFAULT_CONTENT)
    questions = topic_data['questions']
    
    for q in questions:
        u_ans = st.session_state.user_answers.get(q['id'])
        is_corr = (u_ans == q['a'])
        
        with st.expander(f"Q: {q['q']} - {'✅' if is_corr else '❌'}"):
            st.write(f"**Your Answer:** {u_ans}")
            st.write(f"**Correct Answer:** {q['a']}")
            if not is_corr:
                st.warning(f"**Explanation:** {q.get('e', 'No explanation available.')}")
    
    st.markdown("---")
    
    if status == "PASSED":
        st.button("Rank Completed - Return Home", on_click=lambda: st.session_state.update(stage='topic_selection'))
    else:
        st.markdown("### ⚠️ Intervention Required")
        st.write("Your understanding is below the threshold. We will now enter Feynman Mode to clarify weak concepts.")
        if st.button("Enter Feynman Node"):
            start_feynman()
            st.rerun()

def render_feynman():
    st.title("Feynman Re-teaching Phase")
    st.info("We identified gaps in your understanding. Let's break them down simply, as if explaining to a 12-year-old.")
    
    topic_data = CONTENT.get(st.session_state.selected_topic, DEFAULT_CONTENT)
    weak = list(set(st.session_state.weak_concepts))
    
    if not weak:
        # Fallback if logic quirk
        st.write("Reviewing general concepts.")
        weak = topic_data['feynman'].keys()

    for concept in weak:
        # Check if we have a specific Feynman explanation for this Concept ID
        # Concepts are keyed by 'c' in the question (e.g. 'concepts', 'workflow', 'tools')
        # We need to match those to the feynman dict keys.
        
        expl = topic_data['feynman'].get(concept, None)
        
        if expl:
            st.markdown(f"#### Concept: {concept.upper()}")
            st.success(expl)
            st.markdown("---")
        
    if st.button("Ready for Reassessment"):
        start_reassessment()
        st.rerun()

def render_reassessment():
    st.title("Reassessment Quiz")
    st.write("Answer these targeted questions to verify your understanding.")
    
    topic_data = CONTENT.get(st.session_state.selected_topic, DEFAULT_CONTENT)
    weak = list(set(st.session_state.weak_concepts))
    
    # Generate questions dynamically based on errors
    re_questions = []
    re_repo = topic_data.get('reassessment', {})
    
    for w in weak:
        if w in re_repo:
            re_questions.extend(re_repo[w])
    
    # Fallback if too few questions found
    if len(re_questions) < 2:
         # Just grab some valid ones
         all_re = []
         for k in re_repo:
             all_re.extend(re_repo[k])
         if all_re:
             re_questions = all_re[:3]
         
    # Form for submission
    with st.form("reassess_form"):
        responses = []
        for i, q in enumerate(re_questions):
            st.markdown(f"**{i+1}. {q['q']}**")
            val = st.radio("Options", q['opts'], key=f"re_q_{i}")
            responses.append((q, val))
        
        if st.form_submit_button("Submit Reassessment"):
            submit_reassessment(responses)
            st.rerun()

def render_final():
    st.title("Final Evaluation")
    
    status = st.session_state.mastery_status
    score = st.session_state.score
    
    if status == "PASSED_RELEARN":
        st.balloons()
        st.success(f"### Mastery Achieved! Final Score: {score:.1f}%")
        st.write("You successfully repaired your understanding using the Feynman Technique. Great job!")
    else:
        st.error(f"### Module Failed. Final Score: {score:.1f}%")
        st.write("Despite revision, understanding is still not sufficient.")
        st.write("Recommendation: Restart the teaching module from the beginning to build a stronger foundation.")
        
    if st.button("Return to Home"):
        st.session_state.stage = 'topic_selection'
        st.rerun()

# ==========================================
# MAIN ROUTER
# ==========================================

def main():
    if st.session_state.stage == 'topic_selection':
        render_topic_selection()
    elif st.session_state.stage == 'teaching':
        render_teaching()
    elif st.session_state.stage == 'quiz':
        render_quiz()
    elif st.session_state.stage == 'report':
        render_report()
    elif st.session_state.stage == 'feynman':
        render_feynman()
    elif st.session_state.stage == 'reassessment':
        render_reassessment()
    elif st.session_state.stage == 'final':
        render_final()

if __name__ == "__main__":
    main()
