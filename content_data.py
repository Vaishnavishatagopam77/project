
# Content Data for Learning System

CONTENT = {
    1: { # AI
        "context": """
### Artificial Intelligence (AI)

**Artificial Intelligence (AI)** is a broad and comprehensive branch of computer science that is fundamentally defined as the simulation of human intelligence processes by machines, specifically computer systems. It represents a paradigm shift from traditional computing; instead of merely executing pre-written instructions for every specific scenario, AI systems attempt to emulate the cognitive functions associated with the human mind. The ultimate goal is to create systems that can function autonomously and intelligently in complex environments.

The core processes involved in AI are **learning**, **reasoning**, and **self-correction**. These processes allow the system to adapt and improve over time.

*   **Learning**: In the context of AI, learning is defined as the acquisition of information and the rules for using that information. This is distinct from simple data storage or rote memorization. It involves data parsing and pattern recognition to understand the underlying logic and relationships within a dataset. For instance, instead of being told exactly what a "cat" looks like, a learning system analyzes thousands of images to derive the rules that define a cat (ears, whiskers, shape) on its own. It acquires the knowledge necessary to categorize future data.
*   **Reasoning**: This process involves using the rules acquired during the learning phase to reach conclusions. Because the real world is often ambiguous, AI reasoning must be capable of handling different types of conclusions. These can be **approximate** (probabilistic or "best-guess" based on incomplete data) or **definite** (logical certainties derived from strict rules). Reasoning allows the system to navigate new situations by applying its internal logic to solve problems it hasn't explicitly encountered before.
*   **Self-correction**: This is perhaps the most critical aspect for autonomous growth. Self-correction is the ability of the system to monitor its own performance, identify mistakes or suboptimal outcomes, and adjust its internal parameters or rules to improve future performance. This happens without external human intervention. If an AI predicts incorrectly, the self-correction mechanism analyzes the error to ensure it does not repeat the same mistake, effectively "learning from experience."

Significant applications of AI demonstrate these processes in action:

*   **Expert Systems**: These are sophisticated computer programs designed to emulate the decision-making ability of a human expert. Unlike standard procedural code, expert systems rely on a knowledge base of facts and a set of rules (inference engine) to solve complex problems. They are often used in fields like medical diagnosis or financial forecasting, where they can reason through a body of knowledge to provide high-level advice or decisions.
*   **Speech Recognition**: This technology empowers systems to process, interpret, and transcribe human speech. It goes beyond simple recording; the system must understand the nuances of spoken language, accents, and context to convert audio into text or actionable commands. This capability serves as a bridge between human communication and machine execution, enabling voice-activated assistants and dictation tools.
*   **Machine Vision**: This application allows machines to "see" and interpret visual information from the world. It involves capturing digital images or video and analyzing them to identify objects, gauge distances, or detect movement. While a camera captures light, machine vision interprets that data, allowing a computer to understand that a specific collection of pixels represents a "stop sign" or a "pedestrian," similar to how the human visual cortex processes information from the eyes.

In summary, AI is not just about faster processing; it is about the qualitative simulation of cognitive traits—learning from data, reasoning through problems, and correcting errors—to perform tasks that typically require human intelligence.
        """,
        "questions": [
            {"q": "What differentiates AI from traditional computing?", "opts": ["Faster processing speed only", "Simulation of cognitive functions", "Use of keyboards", "Storage capacity"], "a": 2, "c": "definitions", "e": "AI simulates cognitive functions like learning and reasoning, whereas traditional computing executes pre-written instructions."},
            {"q": "The 'Learning' process in AI primarily involves?", "opts": ["Rote memorization", "Acquisition of information and rules", "Hardware upgrades", "Data deletion"], "a": 2, "c": "processes", "e": "Learning is the acquisition of information and the rules for using it, not just memorization."},
            {"q": "Reasoning allows an AI to?", "opts": ["Store more data", "Reach approximate or definite conclusions", "Move physically", "Ignore rules"], "a": 2, "c": "processes", "e": "Reasoning uses rules to reach conclusions, which can be approximate or definite."},
            {"q": "Self-correction enables a system to?", "opts": ["Stop working", "Adjust internal parameters to improve", "Ask for human help", "Forget data"], "a": 2, "c": "processes", "e": "Self-correction allows the system to identify mistakes and adjust parameters without human intervention."},
            {"q": "Expert Systems consist of a knowledge base and?", "opts": ["A camera", "An inference engine (rules)", "A microphone", "A database"], "a": 2, "c": "applications", "e": "Expert systems use a knowledge base of facts and a set of rules (inference engine) to make decisions."},
            {"q": "Speech recognition converts spoken language into?", "opts": ["Images", "Text or commands", "Sound waves", "Video"], "a": 2, "c": "applications", "e": "Speech recognition processes speech to convert it into text or executable commands."},
            {"q": "Machine Vision interprets data from?", "opts": ["Audio files", "Digital images or video", "Text documents", "Spreadsheets"], "a": 2, "c": "applications", "e": "Machine Vision analyzes visual information from digital images or video."},
            {"q": "Approximate conclusions are based on?", "opts": ["Logical certainties", "Probabilities/best-guess", "Strict rules", "Hardware faults"], "a": 2, "c": "processes", "e": "Approximate conclusions are probabilistic or 'best-guesses', often used when data is incomplete."},
            {"q": "Which application emulates a human expert's decision making?", "opts": ["Machine Vision", "Speech Recognition", "Expert Systems", "Data Entry"], "a": 3, "c": "applications", "e": "Expert Systems are specifically designed to emulate the decision-making ability of human experts."},
            {"q": "The ultimate goal of AI described in the text is?", "opts": ["To replace humans entirely", "To function autonomously and intelligently", "To act as a calculator", "To record video"], "a": 2, "c": "definitions", "e": "The text states the goal is to create systems that function autonomously and intelligently."}
        ],
        "feynman": {
            "definitions": "**FEYNMAN:** Imagine a computer that doesn't just do what you type, but thinks like you. It learns from books (data), figures out puzzles (reasoning), and admits when it's wrong (self-correction).",
            "processes": "**FEYNMAN:** Learning is taking notes in class. Reasoning is taking the test using those notes. Self-correction is realizing you got question 5 wrong and studying that chapter again so you ace the final.",
            "applications": "**FEYNMAN:** Expert Systems are like a 'Doctor in a Box'—they know all the medical rules. Speech Recognition is your phone understanding your mumbles. Machine Vision is a robot knowing the difference between a wall and a door."
        },
        "reassessment": {
            "definitions": [
                {"q": "AI simulates?", "opts": ["Mechanical force", "Human intelligence processes", "Chemical reactions", "Historical events"], "a": 2},
                {"q": "Who defines the rules in AI Learning?", "opts": ["The User", "The system derives them from data", "The Hardware Manufacturer", "No one"], "a": 2},
                {"q": "The goal of AI is to?", "opts": ["Replace all jobs", "Function autonomously and intelligently", "Make coffee", "Run slower"], "a": 2}
            ],
            "processes": [
                {"q": "Adjusting internal parameters after an error is?", "opts": ["Learning", "Reasoning", "Self-correction", "Deployment"], "a": 3},
                {"q": "Reasoning involves using rules to?", "opts": ["Forget data", "Reach conclusions", "Delete files", "Cool the CPU"], "a": 2},
                {"q": "Which process handles ambiguity?", "opts": ["Rote memorization", "Approximate reasoning", "Hard coding", "Data deletion"], "a": 2},
                {"q": "Learning is distinct from memorization because it?", "opts": ["Is slower", "Understands underlying logic", "Is manual", "Uses less space"], "a": 2}
            ],
            "applications": [
                {"q": "Interpreting visual data is called?", "opts": ["Speech Recognition", "Machine Vision", "Expert Systems", "Deep Blue"], "a": 2},
                {"q": "An Expert System relies on?", "opts": ["A knowledge base and inference engine", "A camera", "Random guessing", "A microphone"], "a": 1},
                {"q": "Transcribing human speech is valuable for?", "opts": ["Voice-activated assistants", "Image editing", "Data sorting", "Web scraping"], "a": 1}
            ]
        }
    },
    2: { # ML
        "context": """
### Machine Learning (ML)

**Machine Learning (ML)** is a transformative subset of Artificial Intelligence that fundamentally changes how computers solve problems. Rather than relying on explicit programming—where a human developer writes specific code for every possible rule and scenario—ML enables systems to **learn from data**. It focuses on the development of sophisticated algorithms that can adapt, identify complex patterns, and make intelligent decisions based on empirical evidence. The core philosophy is that systems can identify relationships in data that are too complex for humans to hard-code.

ML is generally categorized into three primary types, each defined by how the system interacts with data:

1.  **Supervised Learning**: This is the most common form of ML, comparable to a student learning with a teacher. The system is trained on a dataset that includes **labeled data**. This means every input example in the training set is paired with the correct output (often called the "answer key" or label).
    *   The goal is to map inputs to outputs. For example, if the input is an email and the label is "Spam" or "Not Spam," the model analyzes the features of the email to understand the link.
    *   Once trained, the model can take new, unseen data and predict the correct label based on the patterns it learned from the answer key. It generalizes from the training examples to real-world situations.

2.  **Unsupervised Learning**: This approach deals with **unlabeled data**, which is data that has no historical labels or "answer keys." The system is left to its own devices to find structure.
    *   Without guidance on what is "correct," the algorithm scans the data to identify hidden structures, underlying patterns, or natural groupings.
    *   A common application is clustering, where the system groups similar data points together (e.g., grouping customers by purchasing behavior without knowing ahead of time what the groups are). It is about discovery and organization rather than prediction against a known standard.

3.  **Reinforcement Learning**: This is a dynamic, behavioral type of learning based on **rewards and punishments**.
    *   An independent entity, known as an **agent**, interacts with an environment. The agent makes decisions or performs actions to achieve a goal.
    *   For every action, the agent receives feedback in the form of a mathematical **reward** (for success) or a **penalty/punishment** (for failure or mistakes).
    *   Over time, the agent learns a strategy (policy) to maximize the total cumulative reward. It learns through trial and error, reinforcing the actions that lead to positive outcomes and discarding those that lead to penalties. This is how software learns to play games or control robots.

By leveraging these methods, Machine Learning allows computers to improve their function automatically through experience, making it the engine behind modern predictive analytics and intelligent automation.
        """,
        "questions": [
            {"q": "How does ML differ from explicit programming?", "opts": ["It uses more code", "It enables learning from data instead of hard-coded rules", "It is slower", "It requires no computers"], "a": 2, "c": "basics", "e": "ML enables systems to learn from data rather than relying on explicit programming for every rule."},
            {"q": "Supervised Learning relies on what type of data?", "opts": ["Unlabeled data", "Labeled data (Input + Correct Output)", "Random numbers", "User feedback only"], "a": 2, "c": "types", "e": "Supervised learning uses labeled data, where inputs are paired with the correct output (answer key)."},
            {"q": "What is the 'answer key' in Supervised Learning?", "opts": ["The algorithm", "The input data", "The correct output or label", "The hardware"], "a": 3, "c": "types", "e": "The label associated with the training data acts as the answer key for the model to learn from."},
            {"q": "Unsupervised Learning attempts to find?", "opts": ["Pre-defined answers", "Hidden structures and patterns", "Explicit rules", "Errors in code"], "a": 2, "c": "types", "e": "It deals with unlabeled data to find hidden structures or groupings without an answer key."},
            {"q": "Which is a common application of Unsupervised Learning?", "opts": ["Spam classification", "Clustering/Grouping", "Game playing", "Voice recognition"], "a": 2, "c": "types", "e": "Clustering, such as grouping customers, is a classic unsupervised task (finding groups without labels)."},
            {"q": "Reinforcement Learning involves an agent and?", "opts": ["A teacher", "A static database", "An environment", "A camera"], "a": 3, "c": "types", "e": "In RL, an agent interacts with an environment to learn via feedback."},
            {"q": "Feedback in Reinforcement Learning comes in the form of?", "opts": ["Text messages", "Rewards and punishments", "Labels", "Bug reports"], "a": 2, "c": "types", "e": "The agent receives rewards for success and punishments/penalties for failure."},
            {"q": "The goal of an RL agent is to?", "opts": ["Minimize time", "Maximize cumulative reward", "Use the least memory", "Sort data"], "a": 2, "c": "types", "e": "The agent learns a strategy to maximize the total cumulative reward over time."},
            {"q": "Trial and error is a key characteristic of?", "opts": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Database management"], "a": 3, "c": "types", "e": "RL involves learning through trial and error to determine which actions yield rewards."},
            {"q": "If a system uses an 'answer key' to learn, it is?", "opts": ["Unsupervised", "Reinforcement", "Supervised", "Deep"], "a": 3, "c": "types", "e": "Using labeled data (an answer key) is the definition of Supervised Learning."}
        ],
        "feynman": {
            "basics": "**FEYNMAN:** Traditional programming is like writing a recipe. ML is tasting the food and figuring out the recipe yourself.",
            "types": "**FEYNMAN:** Supervised is a teacher grading your homework (Labels). Unsupervised is finding patterns in the clouds (No labels). Reinforcement is training a puppy with treats (Rewards)."
        },
        "reassessment": {
            "basics": [
                {"q": "ML focuses on?", "opts": ["Static rules", "Algorithms that learn from data", "Hardware design", "Manual coding"], "a": 2},
                {"q": "The core philosophy of ML is?", "opts": ["Humans hard-code everything", "Systems find complex relationships in data", "Random guessing", "Using less power"], "a": 2},
                {"q": "ML enables computers to improve through?", "opts": ["Experience", "Rust", "Magic", "More RAM"], "a": 1}
            ],
            "types": [
                {"q": "Clustering data without labels is?", "opts": ["Supervised", "Unsupervised", "Reinforcement", "Linear"], "a": 2},
                {"q": "Labeled data includes?", "opts": ["Only inputs", "Inputs and correct outputs", "No outputs", "Only errors"], "a": 2},
                {"q": "RL learns policies to maximize?", "opts": ["Errors", "Cumulative reward", "Memory usage", "Heat"], "a": 2},
                {"q": "Training a spam filter usually uses?", "opts": ["Unsupervised Learning", "Supervised Learning", "Reinforcement Learning", "None"], "a": 2},
                {"q": "Finding natural groupings in customer data is an example of?", "opts": ["Clustering", "Regression", "Penalty", "Labeling"], "a": 1},
                {"q": "An RL agent learns via?", "opts": ["Trial and error", "Reading books", "Direct programming", "Ignoring data"], "a": 1},
                {"q": "The 'Teacher' analogy applies to?", "opts": ["Unsupervised Learning", "Supervised Learning", "Reinforcement Learning", "Deep Learning"], "a": 2}
            ]
        }
    },
    3: { # DL
         "context": """
### Deep Learning (DL)

**Deep Learning (DL)** is a specialized and powerful subfield of Machine Learning that is inspired by the structure and function of the human brain. It employs **neural networks** with **many layers**—hence the term "deep"—to learn data representations with high levels of abstraction. While traditional ML often requires human guidance to define features, DL models can learn to recognize these features automatically.

**Key Concepts in Deep Learning:**

*   **Neural Networks**: These are the core computational models of DL. They are designed to mimic the biological connectivity of the human brain. A network consists of interconnected processing nodes, typically referred to as **neurons** or perceptrons. These neurons transmit signal and information, working in concert to solve a problem.

*   **Layers**: A deep learning model is structured like a layer cake, with data passing through multiple stages of processing.
    1.  **Input Layer**: This is the entry point of the network. It receives the raw data (e.g., the pixels of an image or the words of a sentence) and passes it into the system.
    2.  **Hidden Layers**: These are the layers between the input and output. A "deep" network has many hidden layers (sometimes hundreds). It is in these layers that the magic happens. They process the data through increasingly complex **abstractions**. The first layer might find edges; the second might find shapes; the third might find eyes or noses. They extract features without being explicitly told what to look for.
    3.  **Output Layer**: This is the final layer that produces the prediction or decision (e.g., "This image is a cat" or "Stock price will go up").

*   **Representation Learning**: This is a defining capability of DL. In traditional ML, a human often has to tell the computer which features are important (e.g., "look for round shapes"). In Deep Learning, the model possesses the ability to **automatically discover** the representations needed for feature detection or classification from raw data. It learns that "roundness" is important on its own, optimizing its internal filters to identify it.

Deep Learning is the technology behind state-of-the-art advances in fields like image recognition, natural language processing, and autonomous driving, as it can handle vast amounts of unstructured data better than shallow models.
         """,
         "questions": [
             {"q": "Deep Learning is inspired by?", "opts": ["The human liver", "The structure of the human brain", "The digestive system", "Computer chips"], "a": 2, "c": "basics", "e": "DL is inspired by the structure and function of the human brain (neural networks)."},
             {"q": "The 'Deep' in Deep Learning refers to?", "opts": ["Deep philosophical thoughts", "Profound understanding", "The many layers in the network", "Hidden data"], "a": 3, "c": "basics", "e": "It refers to the use of neural networks with many layers (hidden layers)."},
             {"q": "The fundamental processing nodes in a neural network are called?", "opts": ["Pixels", "Neurons", "Synapses", "Cores"], "a": 2, "c": "basics", "e": "The nodes are typically referred to as neurons, mimicking biological neurons."},
             {"q": "Which layer receives the raw data?", "opts": ["Hidden Layer", "Output Layer", "Input Layer", "Processing Layer"], "a": 3, "c": "structure", "e": "The Input Layer is the entry point that receives raw data like pixels or text."},
             {"q": "Where does the processing of complex abstractions happen?", "opts": ["Input Layer", "Hidden Layers", "Output Layer", "Database"], "a": 2, "c": "structure", "e": "The Hidden Layers process data through increasingly complex abstractions."},
             {"q": "The Output Layer is responsible for?", "opts": ["Receiving raw data", "Producing the final prediction", "Finding edges only", "Storing weights"], "a": 2, "c": "structure", "e": "The Output Layer produces the final prediction or decision."},
             {"q": "Representation Learning allows the model to?", "opts": ["Ask for human help", "Automatically discover necessary features", "Run without electricity", "Avoid errors"], "a": 2, "c": "basics", "e": "It allows the model to automatically discover representations needed for feature detection."},
             {"q": "How does DL differ from traditional ML regarding features?", "opts": ["DL requires manual features", "DL automates feature discovery", "DL uses fewer data", "They are identical"], "a": 2, "c": "basics", "e": "In DL, the model learns to identify important features automatically, unlike traditional ML."},
             {"q": "Deep Learning models are particularly good at handling?", "opts": ["Small spreadsheets", "Unstructured data (images, text)", "Simple calculations", "Hardware repairs"], "a": 2, "c": "basics", "e": "DL excels at handling vast amounts of unstructured data like images and text."},
             {"q": "Increasing the number of Hidden Layers generally makes the network?", "opts": ["Shallower", "Deeper", "Slower to learn", "Less accurate"], "a": 2, "c": "structure", "e": "Adding layers makes the network 'deeper' and capable of more complex abstractions."}
         ],
         "feynman": {
             "basics": "**FEYNMAN:** Think of a brain. It has neurons connected by wires. Deep learning is building a fake brain in a computer. 'Deep' just means it has a lot of layers of neurons, like a really thick lasagna.",
             "structure": "**FEYNMAN:** Input is the raw ingredients. Hidden layers are the chefs chopping and cooking (abstracting). Output is the finished dish served to the customer."
         },
         "reassessment": {
             "basics": [
                 {"q": "Neural Networks are meant to mimic?", "opts": ["Biological brains", "Digital clocks", "Calculators", "Cameras"], "a": 1},
                 {"q": "Deep Learning is a subfield of?", "opts": ["Chemistry", "Machine Learning", "Physics", "History"], "a": 2},
                 {"q": "Why is it called 'Deep' Learning?", "opts": ["It is confusing", "It has many layers", "It uses deep holes", "It requires deep sleep"], "a": 2}
             ],
             "structure": [
                 {"q": "Automated feature discovery is called?", "opts": ["Supervised learning", "Representation Learning", "Input processing", "Output generation"], "a": 2},
                 {"q": "Which layer produces the final decision?", "opts": ["Input", "Hidden", "Output", "None"], "a": 3},
                 {"q": "Hidden layers process data into?", "opts": ["Simpler forms", "Complex abstractions", "Raw pixels", "Audio"], "a": 2},
                 {"q": "DL is better than shallow models at handling?", "opts": ["Small data", "Unstructured data", "Simple math", "Hardware"], "a": 2},
                 {"q": "Neurons in a network work in concert to?", "opts": ["Generate heat", "Solve a problem", "Sleep", "Create errors"], "a": 2},
                 {"q": "Traditional ML often requires?", "opts": ["Human guidance for features", "More electricity", "Less data", "Robots"], "a": 1},
                 {"q": "The input layer receives?", "opts": ["Predictions", "Raw data", "Results", "Decisions"], "a": 2}
             ]
         }
    },
    4: { # DS
        "context": """
### Data Science

**Data Science** is a vast, interdisciplinary field that combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful **insights** and knowledge from data. It employs scientific methods, processes, algorithms, and systems to interpret both structured and unstructured data.

The core of Data Science is typically described as a lifecycle or **Pipeline**:

1.  **Collection**: This is the foundational step. Before analysis can begin, raw data must be gathered from various sources. These sources can be diverse, including SQL databases, log files, web scraping, external APIs, or physical sensors (IoT). The quality of the outcome depends heavily on the breadth and relevance of the data collected.

2.  **Cleaning**: Often considered the most time-consuming and unglamorous part of the job (sometimes taking 60-80% of the time), cleaning is vital for accuracy. It involves finding and fixing errors in the data. Tasks include handling **missing values** (deciding to delete or impute them), removing **duplicates**, correcting formatting errors, and filtering out irrelevant noise. Without cleaning, the model suffers from "Garbage In, Garbage Out."

3.  **Exploration (EDA)**: **Exploratory Data Analysis (EDA)** is the detective phase. Here, data scientists visualize the data to understand its main characteristics before applying any formal modeling. They look for trends, patterns, correlations, and outliers. Tools like histograms (for distribution) and scatter plots (for relationships) are essential. EDA helps frame the right questions.

4.  **Modeling**: This is where the math happens. Scientists apply algorithms (often Machine Learning algorithms) to the prepared data to build a model that can predict outcomes or classify new data. This step involves selecting the right algorithm, training it, and tuning its parameters.

5.  **Deployment**: A model is useless if it stays on a laptop. Deployment involves integrating the trained model into a production environment (like a website or mobile app) so it can work on real-world data and provide value to users. It effectively "operationalizes" the science.

Data Science is not just about coding; it is the scientific pursuit of truth hidden within the noise of information.
        """,
        "questions": [
            {"q": "Data Science is primarily concerned with?", "opts": ["Generating random numbers", "Extracting insights and knowledge from data", "Building hardware", "Installing software"], "a": 2, "c": "basics", "e": "Data Science involves extracting insights and knowledge from data using scientific methods."},
            {"q": "Which step in the pipeline involves gathering raw data?", "opts": ["Cleaning", "Modeling", "Collection", "Deployment"], "a": 3, "c": "pipeline", "e": "Collection is the step of gathering raw data from various sources."},
            {"q": "Why is Data Cleaning considered the most time-consuming step?", "opts": ["Computers are slow", "It involves fixing valid data", "It handles missing values, errors, and noise", "It requires no effort"], "a": 3, "c": "pipeline", "e": "Cleaning involves the meticulous task of handling errors, missing values, and noise."},
            {"q": "The phrase 'Garbage In, Garbage Out' emphasizes the importance of?", "opts": ["Collection", "Cleaning", "Modeling", "Deployment"], "a": 2, "c": "pipeline", "e": "It means bad input data (garbage in) leads to bad results (garbage out), highlighting the need for cleaning."},
            {"q": "EDA stands for?", "opts": ["External Database Access", "Exploratory Data Analysis", "Error Data Assessment", "Early Data Acquisition"], "a": 2, "c": "pipeline", "e": "EDA stands for Exploratory Data Analysis."},
            {"q": "What is the main goal of EDA?", "opts": ["To deploy the model", "To visualize and understand data characteristics", "To delete data", "To write the final report"], "a": 2, "c": "pipeline", "e": "EDA is about visualizing data to understand its trends, patterns, and characteristics."},
            {"q": "Modeling involves?", "opts": ["Applying algorithms to predict or classify", "Drawing charts", "Collecting data", "Cleaning text"], "a": 1, "c": "pipeline", "e": "Modeling involves applying algorithms (like ML) to the data to make predictions."},
            {"q": "Deployment refers to?", "opts": ["Deleting the code", "Integrating the model into a real-world environment", "Testing in a notebook", "Ignoring the results"], "a": 2, "c": "pipeline", "e": "Deployment is the integration of the model into a production environment for real use."},
            {"q": "Which tool is mentioned for visualizing relationships in EDA?", "opts": ["Pie chart", "Scatter plot", "List", "Table"], "a": 2, "c": "pipeline", "e": "Scatter plots are mentioned as tools to look for relationships/correlations."},
            {"q": "Data Science combines domain expertise, math, and?", "opts": ["Cooking skills", "Programming skills", "Carpentry", "Painting"], "a": 2, "c": "basics", "e": "It is an interdisciplinary field combining expertise, math/stats, and programming skills."}
        ],
        "feynman": {
            "basics": "**FEYNMAN:** Data Science is being a detective for numbers. You find clues (data), clean them off, look at them with a magnifying glass (EDA), and figure out who did the crime (Insights).",
            "pipeline": "**FEYNMAN:** Imagine cooking. Collection is buying groceries. Cleaning is washing the veggies (boring but needed). EDA is tasting the ingredients. Modeling is crucial cooking. Deployment is serving the meal."
        },
        "reassessment": {
            "basics": [
                {"q": "Data Science outputs?", "opts": ["Insights and knowledge", "More noise", "Hardware products", "Electricity"], "a": 1},
                {"q": "Data Science is an intersection of domain expertise, math, and?", "opts": ["Programming skills", "Music theory", "Botany", "None"], "a": 1}
            ],
            "pipeline": [
                {"q": "Visualizing data happens during?", "opts": ["Collection", "EDA", "Modeling", "Deployment"], "a": 2},
                {"q": "Finding trends and patterns is a goal of?", "opts": ["Cleaning", "EDA", "Collection", "Deployment"], "a": 2},
                {"q": "Cleaning involves fixing?", "opts": ["Hardware glitches", "Duplicate rows and missing values", "Monitor colors", "Network speed"], "a": 2},
                {"q": "Deployment is when the model is?", "opts": ["Deleted", "Operationalized", "Hidden", "Printed"], "a": 2},
                {"q": "Which step ensures 'Garbage Out' doesn't happen?", "opts": ["Cleaning", "Modeling", "Collection", "EDA"], "a": 1},
                {"q": "Collecting data can involve?", "opts": ["Web scraping", "Buying snacks", "Sleeping", "Writing code only"], "a": 1},
                {"q": "Scatter plots are used to see?", "opts": ["Distributions", "Relationships", "Missing values", "Code errors"], "a": 2},
                {"q": "Modeling applies algorithms to?", "opts": ["Raw data", "Prepared/Cleaned data", "Deleted data", "None"], "a": 2}
            ]
        }
    },
    5: { # CV
        "context": """
### Computer Vision

**Computer Vision (CV)** is a fascinating field of Artificial Intelligence that trains computers to **interpret and understand the visual world**. Using digital images from cameras, videos, and deep learning models, machines can accurately identify and classify objects—and then react to what they "see." It essentially gives the computer a pair of eyes and a visual cortex.

To understand CV, one must understand how computers see images:

*   **Pixels**: To a computer, an image is not a picture; it is a grid of numbers. The fundamental unit of this grid is the **pixel** (Picture Element). Each pixel carries numerical information about light intensity or color.
*   **RGB Model**: Most digital screens and CV systems use the RGB color model. An image is composed of 3 stacked distinct color channels: **Red, Green, and Blue**. By mixing the intensity of these three (0-255), we can create the full spectrum of colors perceived by the human eye. A computer "sees" an image as a 3D matrix of these values.

Key Tasks in Computer Vision:

1.  **Object Detection**: This task goes a step beyond simple classification (which just says "there is a cat"). Object Detection asks two questions: "What is in the image?" and **"Where is it?"**.
    *   It identifies objects and draws a **Bounding Box** around them.
    *   This is crucial for things like self-driving cars, which need to detect other cars, pedestrians, and signs simultaneously.

2.  **Segmentation**: This is a more precise and granular task. While detection puts a box around an object, segmentation attempts to trace the object exactly.
    *   It involves **partitioning** an image into segments.
    *   The goal is to assign a label to every single pixel in the image. If detection draws a box around a person, segmentation outlines their exact silhouette, confusing nothing with the background. This pixel-by-pixel precision is vital for medical imaging (detecting tumor shapes) or editing software (removing backgrounds).

Computer Vision is the technology that powers facial recognition, autonomous vehicles, and quality control in manufacturing.
        """,
        "questions": [
             {"q": "What is the primary goal of Computer Vision?", "opts": ["To record audio", "To interpret and understand the visual world", "To clean data", "To display text"], "a": 2, "c": "basics", "e": "CV trains computers to interpret and understand the visual world using digital images."},
             {"q": "To a computer, an image is essentially?", "opts": ["A painting", "A grid of numbers", "A sound wave", "A text file"], "a": 2, "c": "basics", "e": "Computers see images as grids of numbers (pixel values)."},
             {"q": "The fundamental unit of a digital image is?", "opts": ["Voxel", "Pixel", "Byte", "Vector"], "a": 2, "c": "basics", "e": "The pixel (Picture Element) is the fundamental unit."},
             {"q": "The RGB model is composed of which channels?", "opts": ["Red, Green, Black", "Real, Gray, Blue", "Red, Green, Blue", "Red, Gold, Blue"], "a": 3, "c": "basics", "e": "RGB stands for Red, Green, and Blue."},
             {"q": "Object Detection answers which two questions?", "opts": ["Who and Why?", "What and Where?", "When and How?", "What and Why?"], "a": 2, "c": "tasks", "e": "Object Detection identifies 'What is it?' and 'Where is it?'"},
             {"q": "What visual marker does Object Detection use?", "opts": ["A circle", "A Bounding Box", "A single point", "A highlighted pixel"], "a": 2, "c": "tasks", "e": "It draws a Bounding Box around the detected objects."},
             {"q": "Segmentation differs from detection because it?", "opts": ["Is faster", "Partitions the image pixel-by-pixel", "Only finds one object", "Uses no pixels"], "a": 2, "c": "tasks", "e": "Segmentation partitions the image and assigns a label to every single pixel for precise outlining."},
             {"q": "If you need the exact silhouette of an object, you use?", "opts": ["Classification", "Object Detection", "Segmentation", "Compression"], "a": 3, "c": "tasks", "e": "Segmentation outlines the exact shape/silhouette pixel-by-pixel."},
             {"q": "Self-driving cars primarily use Object Detection to?", "opts": ["Listen to horns", "Detect cars and pedestrians", "Read maps", "Connect to WiFi"], "a": 2, "c": "tasks", "e": "They use it to detect and locate obstacles like other cars and pedestrians."},
             {"q": "In RGB, mixing the three colors creates?", "opts": ["Black only", "White only", "The full spectrum of colors", "Grayscale only"], "a": 3, "c": "basics", "e": "Mixing the distinct intensities of Red, Green, and Blue creates the full spectrum of colors."}
        ],
        "feynman": {
            "basics": "**FEYNMAN:** CV is teaching a computer to see. We see a face; the computer sees a spreadsheet of numbers (pixels). RGB is just mixing paint to get the right color.",
            "tasks": "**FEYNMAN:** Object Detection is circling Waldo in a book. Segmentation is carefully cutting Waldo out with scissors so you don't get any background."
        },
        "reassessment": {
            "basics": [
                {"q": "A grid of numbers represents?", "opts": ["An image to a computer", "Sound", "Text", "None"], "a": 1},
                {"q": "Pixel stands for?", "opts": ["Picture Element", "Picture Elect", "Pixar", "None"], "a": 1},
                {"q": "Computer Vision gives a computer?", "opts": ["Ears", "A pair of eyes and visual cortex", "Legs", "An imagination"], "a": 2},
                {"q": "The maximal intensity in standard RGB is?", "opts": ["100", "255", "10", "1"], "a": 2}
            ],
            "tasks": [
                {"q": "Drawing a box around a car is?", "opts": ["Segmentation", "Object Detection", "Cleaning", "RGB"], "a": 2},
                {"q": "Which task answers 'Where is it'?", "opts": ["Classification", "Object Detection", "Compression", "None"], "a": 2},
                {"q": "Partitioning an image into segments is called?", "opts": ["Detection", "Segmentation", "Rendering", "Painting"], "a": 2},
                {"q": "Pixel-by-pixel precision is vital for?", "opts": ["Spam filters", "Medical imaging", "Word docs", "Audio"], "a": 2},
                {"q": "Removing the background requires?", "opts": ["Segmentation", "Detection", "Classification", "None"], "a": 1},
                {"q": "Self-driving cars use detection to see?", "opts": ["Pedestrians", "Air", "Sound", "WiFi"], "a": 1}
            ]
        }
    }
}
