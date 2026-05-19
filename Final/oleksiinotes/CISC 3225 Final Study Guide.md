# Notebook 26
## Decision Tree
A decision tree is a model that makes predictions by asking a sequence of questions.

### Example #1
Human decision tree: a tiny student example
**Features (X)**:
- `hours_studied`
- `attendance_percent`
**Target (y)**:
- `0` = did not pass
- `1` = passed

A very simple human-made rule might be:

```text
If hours_studied >= 5:
    predict pass
else:
    predict fail
```

A decision tree learns rules like this from data.

#### Decision Tree in scikit-learn:
```python
tree_students = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_students.fit(X_students, y_students)

print("Training accuracy:", accuracy_score(y_students, tree_students.predict(X_students)))
```
#### Visualising the tree

Each box is a question.

Read it from top to bottom:
- if the answer is true, go left
- if the answer is false, go right
![DecisionTree Diagram](./Pasted%20image%2020260516115201.png)

> [!NOTE]
> `zip()` is a function to create an iterator that iterates over 2 lists at once even if they are different types.
Print the rules as text
#### Rules as text
This is the most important idea: a tree can be read as rules.
```text
|--- attendance_percent <= 77.50 
| |--- class: 0 
|--- attendance_percent > 77.50 
| |--- class: 1
```
### Example #2
Train a decision tree on digits

For MNIST, the “questions” are about pixel values.

Example question under the hood:

```text
Is pixel 36 <= 2.5?
```

That is harder for humans to interpret than the student example, but it is still a tree of questions.
#### Logistic regression VS Decision Trees
- Logistic regression: weighted combination of inputs
	- Accuracy: 0.96
- Decision tree: sequence of questions
	- Accuracy: 0.71
![Complex DecisionTree Diagram](./Pasted%20image%2020260516121534.png)
#### Decision Tree overfitting
A very deep tree (a lot of rules) may memorize too many details of the training data.
```text
Too simple  -> misses patterns
Too complex -> may memorize details (good only for training data)
```
#### Confusion matrix

The diagonal shows correct predictions. Off-diagonal entries show mistakes.

![Confusion Diagram](./Pasted%20image%2020260516121910.png)

## Final comparison
```text
Logistic Regression -> weighted inputs
KMeans              -> groups by closeness
Decision Tree       -> sequence of questions
```
A decision tree is useful because it is often easier to explain than other models.

# Notebook 27
## Markov Chain
A sequence of events where the probability of moving to the next state depends only on the current state.

### Example #1: Weather Markov Chain
#### Visual Format
Two states Sunny or cloudy, decimals are percent chances to move to a node.
![Weather Diagram](./Pasted%20image%2020260516123029.png)
#### Print Format
```text
{'Sunny': {'Sunny': 0.8, 'Cloudy': 0.2}, 'Cloudy': {'Sunny': 0.4, 'Cloudy': 0.6}}
```
#### Empirical Probability
Markov chains use empirical probability to estimate the likelihood of transitioning between states based purely on historical observation rather than theoretical formulas.
```python
import random

current = "Sunny"

weather = [current]

for _ in range(20):

    if current == "Sunny":
        current = random.choices(
            ["Sunny", "Cloudy"],
            weights=[0.8, 0.2]
        )[0]

    else:
        current = random.choices(
            ["Sunny", "Cloudy"],
            weights=[0.4, 0.6]
        )[0]

    weather.append(current)

print(weather)

from collections import Counter

counts = Counter(weather)

print(counts)
vals = list(counts.values())
print(vals[0]/sum(vals))
print(vals[1]/sum(vals))
```

#### Output:
```text
['Sunny', 'Cloudy', 'Cloudy', 'Cloudy', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Cloudy', 'Sunny', 'Cloudy', 'Cloudy', 'Cloudy', 'Cloudy', 'Sunny', 'Sunny', 'Sunny', 'Sunny']
Counter({'Sunny': 13, 'Cloudy': 8})
0.6190476190476191
0.38095238095238093
```
### Example #2: PageRank
PageRank measures the importance of nodes in a graph based on incoming links.

PageRank is a search engine algorithm that uses a Markov chain to calculate the importance of web pages. It models user behavior as a "random surfer" who hops between pages via hyperlinks. A page's rank represents the probability that a user lands on it.
#### How it works
In the context of PageRank, the mathematical framework relies entirely on Markov chains, which are stochastic, memoryless systems that transition from one state to another.
- States: Individual web pages.
- Transitions: The hyperlinks from one page pointing to another.
- Transition Probabilities: The likelihood of moving from Page $A$ to Page $B$. If Page $A$ has three outgoing links, the probability of transitioning to any single linked page is $\frac{1}{3}$.
- The "Random Surfer" Damping Factor: To prevent the model from getting trapped in "dangling nodes" (pages with no outgoing links) or circular loops, a damping factor (usually $\approx 0.85$) is applied. This simulates a user occasionally getting bored and randomly jumping to any page on the internet.
- Stationary Distribution: The algorithm computes the long-term probabilities of this Markov chain. The pages with the highest probabilities in this steady state are assigned the highest PageRank.
#### Uses and Applications
While popularized by Google to rank search results, the Markov chain-driven principles of PageRank are highly versatile and widely applied to general network analysis.
- Search Engine Optimization (SEO): Webmasters and marketers use PageRank principles to guide their SEO efforts. Understanding how authority flows through links helps in structuring websites and securing high-quality backlinks.
- Network & Node Importance: Identifies "influencers" or central nodes in massive networks.
- Scientific Citations: Evaluates the impact of academic papers based on how many other important papers cite them.
- Recommendation Systems: Suggests products, movies, or music by modeling user behavior as a series of transitions in a chain.
- Natural Language Processing (NLP): Algorithms like _TextRank_ (derived from PageRank) use graphs to score the importance of specific sentences or words in a document for automatic text summarization.
```python
web = {
    "Google": ["Wikipedia", "YouTube"],
    "Wikipedia": ["OpenAI", "Google"],
    "YouTube": ["OpenAI"],
    "OpenAI": ["Wikipedia"]
}
```

```python
current = "Google"

path = [current]

for _ in range(20):

    current = random.choice(web[current])

    path.append(current)

print(path)
```

#### Output:
```text
['Google', 'Wikipedia', 'OpenAI', 'Wikipedia', 'Google', 'Wikipedia', 'OpenAI', 'Wikipedia', 'OpenAI', 'Wikipedia', 'OpenAI', 'Wikipedia', 'Google', 'YouTube', 'OpenAI', 'Wikipedia', 'Google', 'Wikipedia', 'Google', 'YouTube', 'OpenAI']
```
![PageRank Diagram](./Pasted%20image%2020260518001313.png)
```python
pagerank_scores = nx.pagerank(DG)

print(pagerank_scores)
```
#### Output:
```text
{'Google': 0.19825338581235977, 'Wikipedia': 0.3782425942751627, 'YouTube': 0.1217574057248373, 'OpenAI': 0.30174661418764015}
```

# Videos
## Neural Networks (Deep Learning Basics)
**Video:** 3Blue1Brown - But what is a neural network?
### Core Concept 
Inspired by the brain, a neural network is essentially a highly complex mathematical function that processes inputs (like an image of a handwritten digit) through layers of "neurons" to produce a prediction.
### Structure
- **Neurons (Activations):** Nodes that hold numbers (typically between 0 and 1) representing the strength of their activation.
- **Layers:** Networks consist of an **Input Layer** (raw data), **Hidden Layers** (where the pattern recognition happens), and an **Output Layer** (the final prediction).
- **Weights & Biases:** The connections between neurons have **weights** (determining the importance of a connection). Each neuron also has a **bias** (a threshold that must be passed to activate the neuron). "Learning" involves tweaking these thousands (or billions) of knobs.
### Mechanics
- **Weighted Sum:** A neuron takes the outputs from the previous layer, multiplies them by their respective weights, and adds them up along with the bias.
- **Activation Function:** The weighted sum is passed through a function like a **Sigmoid** (squishing values between 0 and 1) or **ReLU** (Rectified Linear Unit) to determine the neuron's final activation.
## K-Means Clustering
**Video:** StatQuest - K-means clustering
### Core Concept
An unsupervised learning algorithm used to group unlabelled data into a specified number of clusters ($K$).
### The Algorithm Steps
1.  **Select $K$:** Decide how many clusters you want to find.
2.  **Initialize:** Randomly place $K$ initial points to act as the first cluster centers.
3.  **Assign:** Measure the distance (typically Euclidean distance) from each data point to the initial centers. Assign each point to the nearest cluster.
4.  **Recalculate:** Calculate the mean (center) of the newly formed clusters. 
5.  **Repeat:** Use the new means as the cluster centers and reassign points. Repeat until the cluster assignments no longer change.
### **Choosing the Best $K$
Use an **Elbow Plot**. Plot the total variance within clusters for different values of $K$. As $K$ increases, variance drops. The optimal $K$ is typically at the "elbow" of the curve, where the reduction in variance starts to level off.
## Large Language Models (LLMs) & Transformers
**Video:** 3Blue1Brown - Large Language Models explained briefly
### Core Concept
LLMs are massive mathematical models with hundreds of billions of parameters. Their fundamental task is simple: **predict the probability of the next word** in a sequence.
### Training Phases
- **Pre-training:** Processing massive amounts of internet text to predict hidden words. An algorithm called **backpropagation** is used to adjust the model's parameters so its predictions get closer to the actual text.
- **RLHF (Reinforcement Learning with Human Feedback):** Fine-tuning the model using human corrections so it behaves as a helpful conversational assistant rather than just a text autocomplete tool.
### The Transformer Architecture
Unlike older models that read text word-by-word, Transformers process entire sequences of text **in parallel**.
- **Attention Mechanism:** The secret sauce of Transformers. It allows the model to look at the surrounding context of a word to refine its meaning (e.g., distinguishing a river "bank" from a financial "bank").
## PageRank Algorithm
**Video:** Arivu - The Algorithm That Made Google Unstoppable: PageRank
### Core Concept
The foundational algorithm behind Google Search. Instead of looking at keyword frequency, it ranks web pages based on the link structure of the internet, treating hyperlinks as "votes of confidence."
### How it Works
Not all votes are equal. A link from a highly authoritative page passes on far more weight than a link from an obscure blog.
- **Random Surfer Model:** Picture a user randomly clicking links forever. A page's "PageRank" is the fraction of time the surfer spends on that page.
- **Damping Factor (~15%):** Accounts for a surfer getting bored and jumping to a completely random page, ensuring they don't get trapped in loops or dead ends.
### Key Insight
Importance is recursive and dynamic. A page is important if other important pages link to it. The entire web can be modeled as a matrix equation to calculate these stable probabilities.
## K-Nearest Neighbors (KNN)
**Video:** Visually Explained - K-nearest Neighbors (KNN) in 3 min
### Core Concept
A highly intuitive algorithm for classification (and regression). It predicts the category of a new, unknown data point based on the majority label of its closest surrounding points.
### The Algorithm
1.  Choose a value for $K$ (number of neighbors to look at).
2.  Calculate the distance (e.g., Euclidean or Manhattan) between the new point and all existing points.
3.  Identify the $K$ closest points.
4.  Assign the most common label among those neighbors to the new point.
### Key Characteristics
- **Lazy Learner:** It doesn't mathematically build or "train" a model in advance. It simply stores the dataset and runs the distance calculations on demand.
- **Curse of Dimensionality:** KNN struggles when there are too many features (high dimensions), because the concept of "distance" becomes less meaningful, negatively impacting accuracy.