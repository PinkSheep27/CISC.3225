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