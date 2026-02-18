
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
Artificial Intelligence (AI) is a field of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence. To understand AI effectively, it is important to explore its foundational concepts such as intelligent agents, search techniques, knowledge representation, reasoning, planning, and learning mechanisms. These concepts together form the backbone of intelligent behavior in machines.

An intelligent agent is a core concept in AI. An intelligent agent is an entity that perceives its environment through sensors and acts upon that environment using actuators. The goal of an intelligent agent is to take actions that maximize its performance based on what it perceives. The agent continuously interacts with its surroundings, making decisions based on incoming information. This interaction between perception and action is what makes the agent intelligent.

Perception in AI refers to the process of gathering raw input from the environment. This input can come from sensors such as cameras, microphones, or other data sources. Perception allows an AI system to understand what is happening around it before deciding on an appropriate action. Without perception, an intelligent agent would not be able to respond meaningfully to its environment.

To make decisions efficiently, AI systems often rely on search algorithms. One such algorithm is A* search. A* search uses heuristics to find the shortest path efficiently between a starting point and a goal. A heuristic is a rule of thumb or an estimation that helps guide the search process toward the goal faster. Instead of exploring all possible paths blindly, heuristics allow the algorithm to focus on the most promising options, saving time and computational effort.

A heuristic is not a guaranteed solution but a practical method that improves efficiency. In AI, heuristics help systems make informed decisions even when complete information is not available. They play a crucial role in problem-solving and search-based tasks.

Another important aspect of AI is knowledge representation. Knowledge representation involves storing facts, rules, and relationships in a structured way so that a machine can understand and reason with them. Proper knowledge representation allows AI systems to draw conclusions, answer questions, and make decisions logically. Without structured knowledge, reasoning would not be possible.

Logic programming is closely related to knowledge representation. Logic programming uses facts and rules to represent knowledge and perform reasoning. Instead of following step-by-step instructions, logic-based systems infer new information based on existing facts and logical rules. This approach is useful in applications that require reasoning and decision-making.

AI also deals with environments where multiple agents compete or interact. In such scenarios, decision-making becomes more complex. The Minimax algorithm is used in adversarial games, where one agent’s success depends on the actions of an opponent. Minimax works by minimizing the possible loss while maximizing the potential gain, assuming the opponent also plays optimally.

Another critical capability of AI systems is planning. Planning in AI means determining a sequence of actions that leads from an initial state to a desired goal. Instead of reacting to the environment randomly, a planning system organizes actions in a logical order to achieve objectives efficiently. Planning is essential for goal-oriented behavior in intelligent agents.

The term Artificial Intelligence itself was coined by John McCarthy, one of the pioneers of the field. His work laid the foundation for AI as a formal academic discipline.

Finally, AI can be categorized into different levels of capability. Strong AI refers to the idea of machines possessing human-level intelligence and consciousness. Unlike systems designed for specific tasks, Strong AI aims to replicate the full range of human cognitive abilities.

In summary, artificial intelligence is built upon interconnected concepts such as intelligent agents, perception, heuristics, search algorithms, knowledge representation, logic programming, planning, and reasoning in competitive environments. Together, these concepts enable machines to perceive, decide, and act intelligently toward achieving goals.
        """,
        "questions": [
            {"id": 1, "q": "What is an Intelligent Agent?", "opts": ["A secret spy", "Entity that perceives and acts", "A database", "A robot arm"], "a": "Entity that perceives and acts", "c": "concepts", "s": 10, "e": "Agents are defined by perception and action cycles."},
            {"id": 2, "q": "A* Search uses heuristics to...", "opts": ["Slow down", "Find shortest path efficiently", "Randomize path", "Encrypt data"], "a": "Find shortest path efficiently", "c": "tools", "s": 9, "e": "Heuristics guide the search to the goal closer, saving time."},
            {"id": 3, "q": "Knowledge Representation involves?", "opts": ["Storing logic/facts", "Storing pixels", "Storing electricity", "Representing colors"], "a": "Storing logic/facts", "c": "concepts", "s": 9, "e": "It structure information so the AI can reason about it."},
            {"id": 4, "q": "The Minimax algorithm is used in?", "opts": ["Sorting", "Adversarial Games", "Painting", "Data cleaning"], "a": "Adversarial Games", "c": "tools", "s": 8, "e": "It minimizes the maximum possible loss, critical for games like Chess."},
            {"id": 5, "q": "Perception in AI refers to?", "opts": ["Feeling emotions", "Gathering raw input", "Moving motors", "Printing logs"], "a": "Gathering raw input", "c": "workflow", "s": 10, "e": "Perception is the input stage (sensors, cameras, microphones)."},
            {"id": 6, "q": "Who coined the term AI?", "opts": ["Turing", "McCarthy", "Musk", "Altman"], "a": "McCarthy", "c": "history", "s": 7, "e": "John McCarthy at the Dartmouth Conference in 1956."},
            {"id": 7, "q": "What is a Heuristic?", "opts": ["A rule of thumb", "A hardware chip", "A coding error", "A database key"], "a": "A rule of thumb", "c": "tools", "s": 9, "e": "Heuristics allow faster decisions by trading optimality for speed."},
            {"id": 8, "q": "Logic Programming uses?", "opts": ["Imperative steps", "Facts and Rules", "Pixels", "Audio waves"], "a": "Facts and Rules", "c": "tools", "s": 8, "e": "It defines 'what' is true, not 'how' to calculate it."},
            {"id": 9, "q": "Planning in AI means?", "opts": ["Scheduling meetings", "Sequence of actions to goal", "Drawing blueprints", "Predicting stock"], "a": "Sequence of actions to goal", "c": "concepts", "s": 9, "e": "Planning constructs a path of states from start to finish."},
            {"id": 10, "q": "Strong AI refers to?", "opts": ["High computational power", "Human-level consciousness", "Durable hardware", "Good at Chess"], "a": "Human-level consciousness", "c": "concepts", "s": 10, "e": "Strong AI (AGI) can apply intelligence to any problem, like a human."}
        ],
        "feynman": {
            "concepts": "FEYNMAN: Think of an agent like a butler. It sees the messy room (perception), knows it should be clean (knowledge), decides to vacuum (reasoning), and does it (action).",
            "tools": "FEYNMAN: Heuristics are like educated guesses. Instead of checking every single street to find your house, you check the neighborhood that looks right first.",
            "workflow": "FEYNMAN: Sense -> Think -> Act. Eyes see the ball, Brain calculates the catch, Hand moves to grab it.",
            "history": "FEYNMAN: McCarthy is the dad of AI. He named the baby in 1956."
        },
        "reassessment": {
            "concepts": [
                {"q": "An agent must have?", "opts": ["Sensors and Actuators", "Legs", "Internet", "Screen"], "a": "Sensors and Actuators"},
                {"q": "A rational agent always?", "opts": ["Wins", "Optimizes performance", "Cheats", "Guesses"], "a": "Optimizes performance"}
            ],
            "tools": [
                {"q": "In A* search, 'h(n)' stands for?", "opts": ["Heuristic", "History", "Hardware", "Height"], "a": "Heuristic"},
                {"q": "Minimax algorithm is best for?", "opts": ["Sorting", "Games (Chess/TicTacToe)", "Cleaning", "Painting"], "a": "Games (Chess/TicTacToe)"}
            ],
            "workflow": [
                {"q": "First step of AI workflow?", "opts": ["Action", "Perception", "Reasoning", "Sleep"], "a": "Perception"},
                {"q": "After perceiving, the agent?", "opts": ["Sleeps", "Decides/Reasons", "Deletes data", "Shuts down"], "a": "Decides/Reasons"}
            ]
        }
    },
    "ML": {
        "context": """
Machine Learning (ML) is a branch of Artificial Intelligence that focuses on enabling systems to learn from data and improve their performance without being explicitly programmed for every task. Instead of following fixed rules, machine learning models identify patterns in data and use those patterns to make predictions or decisions.

One of the most fundamental ideas in machine learning is Supervised Learning. Supervised learning requires labeled data, meaning that each input example comes with a correct output or answer. The model learns by comparing its predictions with the known labels and adjusting itself to reduce errors. Tasks such as predicting house prices or identifying spam emails commonly use supervised learning because the correct outcomes are known beforehand.

A common task in supervised learning is Regression. Regression is used when the output is a continuous numerical value. For example, predicting house prices involves estimating a numeric value based on input features like size or location. Since prices are not categories but continuous numbers, this task falls under regression rather than classification.

Another important supervised task is Classification. Classification involves assigning inputs into predefined categories or classes. Spam filtering is a classic classification task, where emails are categorized as either spam or non-spam (ham). Classification problems focus on deciding which class an input belongs to rather than predicting a numeric value.

In contrast to supervised learning, Unsupervised Learning works with unlabeled data. In this case, the model does not have access to correct answers. Instead, it tries to discover hidden patterns or structures in the data on its own. A common unsupervised task is clustering, such as grouping customers based on similar behavior without predefined labels. This helps in understanding the natural organization of data.

Another major learning paradigm is Reinforcement Learning. Reinforcement learning does not rely on labeled data or clustering. Instead, it uses rewards and penalties to guide learning. An agent interacts with an environment, takes actions, and receives feedback in the form of rewards or penalties. Over time, the agent learns to maximize its cumulative reward by choosing better actions.

Machine learning models must generalize well to new, unseen data. A major challenge that affects generalization is Overfitting. Overfitting occurs when a model memorizes the training data instead of learning meaningful patterns. Such a model performs very well on training data but fails when tested on new data. Poor performance on unseen data is a strong indicator of overfitting.

To prevent overfitting and ensure reliability, the machine learning workflow includes several important steps. One of the earliest steps is Data Preprocessing. Data preprocessing involves cleaning and normalizing data so that it can be effectively used for training. Real-world data is often messy, incomplete, or inconsistent, so preprocessing is essential for building accurate models.

After data preprocessing and training, the next critical step is Evaluation. Evaluation measures how well a trained model performs on unseen data. Metrics such as Accuracy are commonly used for this purpose. Accuracy measures the percentage of correct predictions made by the model out of all predictions. It provides a simple and intuitive way to assess model performance, especially in classification tasks.

Once a model has been evaluated and shown to perform well, it can move toward deployment. However, evaluation is a crucial checkpoint to ensure that the model works as expected before being used in real-world scenarios.

In summary, machine learning involves different learning paradigms such as supervised, unsupervised, and reinforcement learning. Concepts like regression, classification, overfitting, data preprocessing, evaluation, and accuracy form the foundation of effective machine learning systems. Understanding these concepts helps in building models that not only perform well on training data but also generalize successfully to new data.
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
            "concepts": "FEYNMAN: Supervised is flashcards. Unsupervised is sorting blocks by color without knowing names. Overfitting is memorizing the practice test but failing the real one.",
            "workflow": "FEYNMAN: Gather ingredients -> Chop -> Cook -> Taste test -> Serve.",
            "tools": "FEYNMAN: Regression guesses a number (Height). Classification guesses a name (Boy/Girl)."
        },
        "reassessment": {
            "concepts": [
                {"q": "If data has no target label, use?", "opts": ["Supervised", "Unsupervised", "Regression", "Checking"], "a": "Unsupervised"},
                {"q": "Reinforcement learning relies on?", "opts": ["Rewards/Penalties", "Teachers", "Books", "Videos"], "a": "Rewards/Penalties"}
            ],
            "workflow": [
                {"q": "Cleaning missing values happens in?", "opts": ["Deployment", "Preprocessing", "Evaluation", "None"], "a": "Preprocessing"},
                {"q": "Splitting data into Training and Test sets is for?", "opts": ["Validation", "Formatting", "Coloring", "Printing"], "a": "Validation"}
            ],
            "tools": [
                {"q": "Predicting if it will rain (Yes/No)?", "opts": ["Regression", "Classification", "Clustering", "Summarization"], "a": "Classification"},
                {"q": "To predict a house price (Value)?", "opts": ["Regression", "Classification", "Clustering", "K-Means"], "a": "Regression"}
            ]
        }
    },
    "DL": {
        "context": """Deep Learning is a specialized area of Artificial Intelligence that focuses on learning complex patterns from data using neural networks with many layers. It is inspired by the human brain, particularly the way biological neurons and synapses work together to process information. Just as the brain learns from experience, deep learning models learn from data through repeated exposure and adjustment.

At the heart of deep learning are Artificial Neural Networks. These networks are made up of layers of interconnected units called neurons. Each neuron receives inputs, processes them, and passes the result to the next layer. A deep learning model typically consists of three main types of layers: the input layer, one or more hidden layers, and the output layer. The hidden layers are located between the input and output layers, and they are responsible for performing intermediate computations and extracting meaningful features from the data.

One of the most important processes in deep learning is training, where the model learns how to make accurate predictions. During training, the model processes data and produces an output. This output is then compared with the correct answer, and the difference between them is calculated using a loss function. Loss calculation measures the error or difference between the predicted output and the actual target. A lower loss indicates better performance.

To reduce this error, deep learning models use a learning algorithm called backpropagation. Backpropagation is used to update weights and enable learning. It works by propagating the error backward from the output layer to the earlier layers. Based on this error, the model adjusts the weights of connections between neurons so that future predictions become more accurate. This process is repeated many times during training.

A complete pass through the entire training dataset is known as an epoch. During one epoch, the model sees every training example once. Training usually involves multiple epochs so that the model can gradually improve its performance. With each epoch, the model ideally becomes better at capturing the underlying patterns in the data.

Deep learning uses different types of neural network architectures depending on the nature of the data. Convolutional Neural Networks (CNNs) are best suited for images and video. CNNs use special layers called convolutional layers that are designed to capture spatial patterns such as edges, shapes, and textures. These spatial hierarchies make CNNs highly effective for visual data.

In contrast, Recurrent Neural Networks (RNNs) are designed to handle sequential data. RNNs differ from CNNs because they have memory and loops. This means they can retain information from previous steps and use it when processing the next input. This ability makes RNNs suitable for data where order matters, such as sequences.

Another powerful deep learning architecture is the Transformer. Transformers rely on an attention mechanism rather than recurrence or convolution. Attention allows the model to focus on the most important parts of the input when processing information. This mechanism enables the model to weigh different elements differently based on their relevance, improving learning efficiency.

Deep learning models also rely on activation functions to introduce non-linearity into the network. An example of an activation function is ReLU (Rectified Linear Unit). Activation functions help the network learn complex relationships that cannot be captured by simple linear transformations.

Deep learning systems are known for their high performance, but they also have significant requirements. Deep learning requires large amounts of data and computational power. Because these models have many layers and parameters, they need extensive data to learn effectively and powerful hardware to train efficiently.

In summary, deep learning is inspired by the human brain and is built upon multi-layered neural networks. Key concepts include hidden layers, loss calculation, backpropagation, epochs, and activation functions. Different architectures such as CNNs, RNNs, and Transformers are used for different types of data. While deep learning is powerful, it requires large datasets and substantial computation to function effectively.""",
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
            "concepts": "FEYNMAN: A Neural Net is a team of detectives. Backprop is the boss correcting them.",
            "workflow": "FEYNMAN: Guess -> Check Score -> Learn -> Repeat.",
            "tools": "FEYNMAN: CNN = Eyes. RNN = Memory. Transformers = Speed Readers."
        },
        "reassessment": {
            "concepts": [
                {"q": "Weights are adjusted during?", "opts": ["Forward Prop", "Backpropagation", "Input", "Testing"], "a": "Backpropagation"},
                {"q": "Dropout is used to fix?", "opts": ["Overfitting", "Underfitting", "Slow speed", "Low battery"], "a": "Overfitting"}
            ],
            "workflow": [
                {"q": "Forward propagation produces?", "opts": ["Prediction", "Error", "Weights", "Gradients"], "a": "Prediction"},
                {"q": "One complete pass through the training data is?", "opts": ["Epoch", "Batch", "Step", "Cycle"], "a": "Epoch"}
            ],
            "tools": [
                {"q": "To process a sentence, use?", "opts": ["CNN", "RNN", "K-Means", "Linear Regression"], "a": "RNN"},
                {"q": "Which network is best for images?", "opts": ["CNN", "RNN", "MLP", "Transformer"], "a": "CNN"}
            ]
        }
    },
    "DS": {
        "context": """Data Science is a discipline that focuses on extracting meaningful insights from data. It combines data handling, analysis, visualization, and domain understanding to support informed decision-making. To work effectively with data, it is essential to follow a structured workflow and apply proper tools and principles.

One of the most important early steps in data science is Exploratory Data Analysis (EDA). EDA stands for Exploratory Data Analysis. It is the process of summarizing, understanding, and visualizing the main characteristics of a dataset. Instead of immediately building models, EDA helps analysts explore patterns, trends, and anomalies in the data. This step allows a deeper understanding of what the data contains and how it behaves.

A commonly used workflow in data science is the OSEMN framework, which stands for Obtain, Scrub, Explore, Model, and Interpret. The first step in OSEMN is Obtain, which involves collecting or querying data from various sources. Without data, no analysis is possible, so obtaining relevant and reliable data is the foundation of the entire process.

Once data is obtained, it often needs cleaning. This is where the principle “Garbage In, Garbage Out” becomes critical. This phrase means that bad data leads to bad results. If the input data is inaccurate, incomplete, or inconsistent, the output of analysis or models will also be unreliable. Therefore, ensuring data quality is essential.

Data Cleaning, also known as data scrubbing, involves fixing missing values, correcting errors, and ensuring data consistency. Raw data frequently contains duplicates, incorrect entries, or missing information. Cleaning the data ensures that constraints and validity rules are met, which improves the accuracy of subsequent analysis and modeling.

After cleaning, analysts often rely on tools to explore and manipulate data efficiently. One such tool is SQL (Structured Query Language). SQL is used for querying databases, allowing analysts to retrieve, filter, and aggregate data stored in relational databases. SQL plays a crucial role in the Obtain and Scrub stages of the data science workflow.

Another widely used tool is Pandas, a Python library designed for data manipulation. Pandas provides powerful data structures that make it easier to clean, transform, and analyze datasets. It allows analysts to handle large datasets efficiently and perform operations such as filtering, grouping, and merging data.

Visualization is another key aspect of data science. Visualization is essential for interpretation and communication. Humans understand visual patterns much faster than raw tables of numbers. Charts, graphs, and plots help reveal trends, relationships, and outliers that may not be obvious otherwise.

A heatmap is one type of visualization commonly used in data analysis. Heatmaps represent data values using color intensity in two dimensions. They are particularly useful for identifying patterns, correlations, or areas of high and low values within a dataset. By visualizing magnitude through color, heatmaps make complex data easier to interpret.

Beyond tools and techniques, data science also requires domain expertise. In data science, domain expertise means understanding the specific industry or subject area related to the data. Without domain knowledge, it is difficult to ask the right questions, interpret results correctly, or draw meaningful conclusions. Domain understanding adds context and relevance to data analysis.

The final stages of the OSEMN framework involve Modeling and Interpretation. Modeling applies statistical or computational techniques to learn from data, while interpretation focuses on explaining results and communicating insights clearly. Notably, “Negotiate” is not a step in OSEMN, reinforcing the importance of following a well-defined lifecycle.

In summary, data science relies on a structured workflow, starting with obtaining data and emphasizing data quality through cleaning and validation. Exploratory Data Analysis, visualization, and tools such as SQL and Pandas help analysts understand and manipulate data effectively. Combined with strong domain expertise, these components enable meaningful interpretation and reliable insights from data.""",
        "questions": [
            {"id": 1, "q": "What does EDA stand for?", "opts": ["Exploratory Data Analysis", "External Data Access", "Error Detection Algorithm", "Easy Data Action"], "a": "Exploratory Data Analysis", "c": "concepts", "e": "EDA is the process of summarizing and visualizing main characteristics."},
            {"id": 2, "q": "The first step in OSEMN is?", "opts": ["Model", "Obtain", "Scrub", "Interpret"], "a": "Obtain", "c": "workflow", "e": "You must obtain data (collect/query) before you can do anything else."},
            {"id": 3, "q": "Garbage In, Garbage Out refers to?", "opts": ["Recycling", "Bad data leads to bad results", "Deleting files", "Formatting hard drives"], "a": "Bad data leads to bad results", "c": "concepts", "e": "Quality of output is determined by quality of input data."},
            {"id": 4, "q": "SQL is used for?", "opts": ["Plotting", "Querying Databases", "Web Design", "Training Neural Nets"], "a": "Querying Databases", "c": "tools", "e": "SQL (Structured Query Language) retrieves data from relational DBs."},
            {"id": 5, "q": "Data Cleaning involves?", "opts": ["Wiping the screen", "Fixing missing/error values", "Deleting all data", "Compressing files"], "a": "Fixing missing/error values", "c": "workflow", "e": "Scrubbing data ensures constraints and validity are met."},
            {"id": 6, "q": "Visualization is key for?", "opts": ["Interpretation/Communication", "Storage", "Encryption", "Networking"], "a": "Interpretation/Communication", "c": "tools", "e": "Humans understand visual patterns faster than raw tables of numbers."},
            {"id": 7, "q": "Pandas is a tool for?", "opts": ["Zoos", "Data Manipulation", "3D Printing", "Sound Editing"], "a": "Data Manipulation", "c": "tools", "e": "Pandas is the primary Python library for data analysis."},
            {"id": 8, "q": "A Heatmap is a type of?", "opts": ["Database", "Visualization", "Server", "Algorithm"], "a": "Visualization", "c": "tools", "e": "Heatmaps show magnitude of phenomena as color in two dimensions."},
            {"id": 9, "q": "In DS, Domain Expertise means?", "opts": ["Knowing the specific industry", "Owning a website", "Being an admin", "Knowing math"], "a": "Knowing the specific industry", "c": "overview", "e": "You need to understand the subject to interpret the data."},
            {"id": 10, "q": "Which is NOT a step in OSEMN?", "opts": ["Obtain", "Scrub", "Negotiate", "Model"], "a": "Negotiate", "c": "workflow", "e": "Negotiate is not part of the lifecycle."}
        ],
        "feynman": {
            "concepts": "FEYNMAN: Cleaning data is like washing vegetables. EDA is checking ingredients before cooking.",
            "workflow": "FEYNMAN: Get -> Clean -> Look -> Math -> Explain.",
            "tools": "FEYNMAN: SQL is the librarian. Visualization is the picture."
        },
        "reassessment": {
            "concepts": [
                {"q": "Why clean data?", "opts": ["To save space", "To ensure accuracy", "To make it pretty", "To hide it"], "a": "To ensure accuracy"},
                {"q": "Garbage In, Garbage Out means?", "opts": ["Recycling", "Bad input = Bad output", "Deleting files", "Formatting"], "a": "Bad input = Bad output"}
            ],
            "workflow": [
                {"q": "Last step of DS workflow?", "opts": ["Scrub", "Interpret/Act", "Obtain", "Model"], "a": "Interpret/Act"},
                {"q": "First step of OSEMN framework?", "opts": ["Obtain", "Model", "Scrub", "Explore"], "a": "Obtain"}
            ],
            "tools": [
                {"q": "Best chart for correlation?", "opts": ["Pie Chart", "Scatter Plot", "Bar Chart", "List"], "a": "Scatter Plot"},
                {"q": "SQL is used to?", "opts": ["Design UI", "Query Databases", "Train models", "Draw charts"], "a": "Query Databases"}
            ]
        }
    },
    "CV": {
        "context": """Computer Vision (CV) is a field of Artificial Intelligence that enables machines to interpret and understand visual information from images and videos. At its foundation, computer vision treats visual data not as pictures but as numerical information that can be processed and analyzed by algorithms.

An image is essentially a grid of numbers. Each number represents a pixel value, which corresponds to the intensity of light at a specific position in the image. In grayscale images, pixel values usually range from 0 to 255. A pixel value of 0 represents black, while a value of 255 represents white. Intermediate values correspond to different shades of gray. This numerical representation allows computers to apply mathematical operations to images.

Before images are analyzed, they often go through preprocessing. Preprocessing in computer vision includes operations such as resizing images and converting them to grayscale. These steps standardize input data so that models receive consistent image dimensions and formats. Preprocessing helps reduce complexity and improves the performance of computer vision algorithms.

One fundamental operation in computer vision is edge detection. Edge detection looks for sharp changes in brightness or intensity within an image. These changes usually occur at object boundaries. By detecting edges, computer vision systems can identify shapes and structures within images, which is crucial for understanding visual content.

Computer vision tasks vary depending on the level of detail required. Object Detection is a task where the system identifies both the class and the location of objects within an image. This means the system not only recognizes what the object is but also determines where it appears, often using bounding boxes. Object detection is more detailed than simple image classification, which only assigns a single label to the entire image.

A well-known object detection approach is YOLO, which stands for “You Only Look Once.” This name reflects its design philosophy of processing an image in a single pass to detect objects quickly. YOLO is known for its speed because it predicts object classes and locations simultaneously instead of using multiple stages.

Another important computer vision task is segmentation. Segmentation involves classifying each individual pixel in an image. Instead of assigning one label to the entire image or bounding boxes to objects, segmentation assigns a class to every pixel. This provides a detailed understanding of image regions and object boundaries.

Computer vision systems often rely on specialized libraries to perform image processing tasks efficiently. OpenCV is one such library. OpenCV is a computer vision library that provides tools for image processing, feature detection, and video analysis. It is widely used because it offers optimized functions for handling visual data.

To improve the robustness of computer vision models, data augmentation is commonly used. Data augmentation helps with model generalization by artificially increasing the diversity of training data. This is done by applying transformations such as rotation, flipping, or scaling. By seeing varied versions of the same data, models become better at handling real-world variations.

Computer vision plays a critical role in many advanced systems. For example, self-driving cars use computer vision to detect lanes and obstacles on the road. Visual information helps these systems understand road structure, identify objects, and make safe navigation decisions.

Throughout the computer vision pipeline, the quality of input data and preprocessing steps greatly influence the final outcome. Images must be represented correctly as numerical grids, cleaned, and standardized before higher-level tasks like detection or segmentation are applied.

In summary, computer vision is built on the idea that images are numerical data composed of pixels. Key concepts include pixel values, preprocessing, edge detection, object detection, segmentation, and data augmentation. Tools like OpenCV support these processes, while approaches such as YOLO enable fast object detection. Together, these components allow machines to extract meaningful information from visual data and perform tasks such as navigation, recognition, and scene understanding.""",
        "questions": [
            {"id": 1, "q": "A pixel value of 0 usually means?", "opts": ["White", "Black", "Red", "Transparent"], "a": "Black", "c": "concepts", "e": "In grayscale, 0 is black, 255 is white."},
            {"id": 2, "q": "Object Detection finds?", "opts": ["Class only", "Class and Location", "Pixel mask", "Resolution"], "a": "Class and Location", "c": "concepts", "e": "It labels objects and defines their location (Bounding Box)."},
            {"id": 3, "q": "YOLO stands for?", "opts": ["You Only Live Once", "You Only Look Once", "Yellow Owl", "Yield Output"], "a": "You Only Look Once", "c": "tools", "e": "Refers to fast, single-pass processing."},
            {"id": 4, "q": "Segmentation is?", "opts": ["Classifying the whole image", "Classifying pixels", "Cropping", "Zooming"], "a": "Classifying pixels", "c": "concepts", "e": "Assigns a class to every single pixel."},
            {"id": 5, "q": "Edge Detection looks for?", "opts": ["Colors", "Sharp brightness changes", "Smooth areas", "Faces"], "a": "Sharp brightness changes", "c": "tools", "e": "Discontinuities in intensity define boundaries."},
            {"id": 6, "q": "Preprocessing in CV includes?", "opts": ["Resizing/Grayscale", "Deleting camera", "Compiling code", "Networking"], "a": "Resizing/Grayscale", "c": "workflow", "e": "Standardizes input for models."},
            {"id": 7, "q": "OpenCV is?", "opts": ["A VR headset", "A CV Library", "A camera brand", "A robot"], "a": "A CV Library", "c": "tools", "e": "Essential library for image processing."},
            {"id": 8, "q": "Data Augmentation helps?", "opts": ["Reduce dataset size", "Model Generalization", "Make images smaller", "Blur images"], "a": "Model Generalization", "c": "tools", "e": "Makes model robust to variations like rotation."},
            {"id": 9, "q": "An image is essentially?", "opts": ["A list of words", "A grid of numbers", "A sound wave", "A vector"], "a": "A grid of numbers", "c": "concepts", "e": "Matrices of pixel intensity values."},
            {"id": 10, "q": "Self-driving cars use CV to?", "opts": ["Listen to radio", "Detect lanes/obstacles", "Cool the engine", "Check oil"], "a": "Detect lanes/obstacles", "c": "overview", "e": "Primary sensor modality for road navigation."}
        ],
        "feynman": {
            "concepts": "FEYNMAN: Pixels are Lite-Brite pegs. Detection is boxing; Segmentation is tracing exact edges.",
            "workflow": "FEYNMAN: See -> Clean -> Outline -> Recognize -> Act.",
            "tools": "FEYNMAN: OpenCV is the Swiss Army Knife for eyes."
        },
        "reassessment": {
            "concepts": [
                {"q": "RGB stands for?", "opts": ["Red Green Blue", "Real Good Byte", "Radio Graphic Base", "Red Gray Black"], "a": "Red Green Blue"},
                {"q": "Pixel values usually range from?", "opts": ["0-10", "0-100", "0-255", "0-1000"], "a": "0-255"}
            ],
            "workflow": [
                {"q": "Removing noise happens in?", "opts": ["Acquisition", "Preprocessing", "Action", "None"], "a": "Preprocessing"},
                {"q": "Data Augmentation helps in?", "opts": ["Reducing size", "Generalization", "Cropping", "Saving electricity"], "a": "Generalization"}
            ],
            "tools": [
                {"q": "Annotating every pixel is?", "opts": ["Detection", "Segmentation", "Classification", "Regression"], "a": "Segmentation"},
                {"q": "Object Detection finds?", "opts": ["Only Class", "Class & Location", "Nothing", "Colors"], "a": "Class & Location"}
            ]
        }
    }
}
