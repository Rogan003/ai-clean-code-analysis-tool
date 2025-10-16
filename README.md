# AI Clean Code Analysis Tool

A proposal for an AI model that assesses the quality, readability, and maintainability of Java classes and methods using a hybrid approach that combines heuristics, K-Nearest Neighbors (KNN), and Convolutional Neural Networks (CNN). The tool aims to highlight code sections that violate Clean Code principles (similar to the ones described by Robert C. Martin in his book "Clean Code") and to suggest where improvements may be needed.

---

## 1) Project Title

Automated analysis of Java classes and methods according to Clean Code principles using a hybrid approach (Heuristics + KNN + CNN).

---

## 2) Problem Definition

Modern software development requires code that is both maintainable and readable to reduce technical debt, streamline team collaboration, and speed up bug discovery. This project proposes a system that analyzes Java classes and methods and classifies them into one of three readability/maintainability categories:

- Green — Good
- Yellow — Changes recommended
- Red — Changes required

The output should emphasize classes—and especially methods—that break Clean Code principles or could be improved. The model will combine heuristics, KNN-based classification, and deep learning.

---

## 3) Motivation

Beginners often receive pull request (PR) feedback pointing out code that is not easily maintainable or readable. The author’s personal experience is that such feedback, while valuable, would have been accelerated by an automated assistant. A tool like this can help:

- Beginners quickly spot critical code sections and learn Clean Code principles.
- Professional teams obtain fast assessments of critical code areas with regard to readability and maintainability.

---

## 4) Datasets

Initial data acquisition ideas:

1. Leverage existing public datasets from similar projects. If an ideal match is not found, adapt the closest datasets to the project’s needs. Example public datasets:
   - https://huggingface.co/datasets/se2p/code-readability-krod
   - https://huggingface.co/datasets/se2p/code-readability-merged
2. Collect publicly available Java projects from GitHub and label them manually and via PR comments. Ideally involve a few experienced developers and the author’s own previously reviewed code. Existing AI models (e.g., ChatGPT) may assist with initial labeling, but human review will validate the labels.

---

## 5) Data Preprocessing

Planned preprocessing steps:

1. Split data into train/validation/test sets.
2. Parse Java code into classes and methods via suitable libraries.
3. Compute selected heuristic metrics (e.g., number of methods per class, lines of code per method, number of parameters, nesting depth).
4. Tokenize Java classes and methods into code tokens and transform them into sequences for the CNN component.

---

## 6) Methodology (Hybrid Approach)

The system combines three components that contribute to the final score (not necessarily with equal weights):

1. Heuristic layer — Rules derived from Clean Code guidelines (e.g., function length limits, parameter counts, nesting depth thresholds).
2. KNN classifier — Classifies code segments based on numeric features (e.g., number of parameters, length, nesting, etc.).
3. CNN model — Consumes tokenized code sequences to detect patterns of poor style and readability.

The outputs of these components will be combined into a final assessment.

### Heuristics used for classes
- Name: 3+ && <11, <20, 21+ || 2-
- Property names length: 3+ && <10, <18, 19+ || 2-
- Property names chars: no special chars, 1 special char, 2+ special chars
- Number of public methods (excluding getters and setters): <5, <7, 8+
- Number of variables: <7, <10, 11+
- All methods average score: <1.5, <2.3, 2.31+
- Total amount of comment chars: <150, <300, 301+
- Simplified check of class cohesion (how many methods use how many variables)
- Class name, var names in camelCase: yes, some, no

### Heuristics used for methods
- Name: 3+ && <11, <20, 21+ || 2-
- Parameter/local var names length: 3+ && <8, <13, 14+ || 2-
- Parameter/local var names chars: no special chars, 1 special char, 2+ special chars
- Length: <10, <30, 31+
- Number of parameters: <2, <4, 5+
- Total amount of comment chars: <50, <100, 101+
- Deepest indentation level: <3, <4, 5+
- Longest line: <80, <110, 111+
- Cyclomatic complexity: <O(n^2), <O(n^3), O(n^3)+
- Number of return statements: <2, <3, 4+
- Method name, parameter names, local var names in camelCase: yes, some, no

---

## 7) Evaluation

Evaluation will compare predictions against a held-out, labeled test set (from the train/val/test split). The primary metric is classification accuracy per the three categories (Good / Changes recommended / Changes required). Special attention will be given to strongly misclassified cases (e.g., ground truth “Good” predicted as “Changes required,” or vice versa) to analyze failure modes and improve the models.

---

## 8) Technologies

- Python AI/ML libraries: scikit-learn, PyTorch, TensorFlow (to be finalized).
- Python libraries for parsing Java (to be selected based on suitability and ecosystem support).

---

## 9) References and Related Work

- Robert C. Martin, "Clean Code: A Handbook of Agile Software Craftsmanship."
- Andrew Hunt and David Thomas, "The Pragmatic Programmer: Your Journey to Mastery."
- Additional AI/ML literature will be explored as the project evolves. While many modern tools leverage LLMs, a directly matching publication has not yet been identified. A related example that overlaps part of this idea:
  - https://github.com/ehtishamDev/Automated-Code-Quality-Classification-with-Convolutional-Neural-Networks-and-NLP

---

## Project Status

This document captures the initial project proposal and design intent. The exact datasets, parsing libraries, model architectures, and weighting of hybrid components are subject to iteration based on experiments and findings.

---

## Major flaws

- When talking about variable or class or method names, we can't see if these names are meaningful or not.
- We can't see if some method or class do one or multiple things.
- We don't look at coupling with other classes. Only the specific class and its methods.
- We don't catch code duplication necessarily through heuristics.

## Potential Next Steps (Suggestions)

- Finalize dataset sources and labeling strategy.
- Evaluate candidate Java parsing libraries and implement robust parsing to class/method granularity.
- Define and validate heuristic rules grounded in Clean Code principles.
- Engineer numeric features for KNN and prototype the CNN tokenization pipeline.
- Establish evaluation metrics and baselines; create error analysis tooling.
- Explore augmentation with LLM-based signals as an important fourth component.

---

## Author

Veselin Roganović (SV 36/2022), this project was done as a university project