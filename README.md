# UML model extraction from text

The project aims to use automatic learning tools to extract the model from textual specifications. More specifically, we want to use the machine learning architectures developed in the context of natural language processing tasks to extract the concepts present and the links between them. We make the following assumptions. First, the tools developed for detecting entities allow us to discover the concepts. Second, the tools developed for resolving coreferences and relation classification allow us to discover the links between these concepts. Collecting all this information makes it possible to produce a business model in the form of a UML class diagram.
The use of machine learning to produce a domain model has not yet been realized. In this sense, it is a new task in machine learning. As with any new task, no benchmark, model, or even dataset can be used as a basis. The task resolution mode is unknown. The contribution of this project is, therefore, to propose a way of solving this task involving 1) the development of a data set composed of specifications explicitly annotated for the resolution of this task, 2) the development of a model of a resolution intended to analyze the sequential structure of the sentences which constitute the specifications and to jointly produce a set of results in the field of the identification of concepts, resolution of coreferences and classification of relations necessary for the resolution of the task and 3) a reference for possible future studies on the subject.

## The dataset
The dataset comprises 20 documents with 178 to 1,537 tokens each for 10,604 tokens. It is divided into a training set composing 14 documents (8,258 tokens), i.e., 70% of all documents, a validation set containing three documents (1,572 tokens), i.e., 15% of the set of documents, and a test set also including three documents (784 tokens). 

<b>Acknowledgments:</b> The texts of the example specifications in this dataset are exercises taken from software engineering (UML) courses or databases (entity-relationship model) found on the internet. We, therefore, thank their authors here. Most of the texts were in French, so we translated them into English. We have annotated the specifications with our proposed annotation.

<b>Note!<b> Add GloVe 6B in data folder

## The model and validation results
See the references below.

## References
Rigou, Y., Khriss, I. (2023). A Deep Learning Approach to UML Class Diagrams Discovery from Textual Specifications of Software Systems. In: Arai, K. (eds) Intelligent Systems and Applications. IntelliSys 2022. Lecture Notes in Networks and Systems, vol 543. Springer, Cham. https://doi.org/10.1007/978-3-031-16078-3_49. https://link.springer.com/chapter/10.1007/978-3-031-16078-3_49
Rigou, Y. (2022). Une approche de découverte des diagrammes de classes UML par l’apprentissage profond. Mémoire de Maîtrise, Rimouski, QC, Canada. Directeur de mémoire : Khriss, I. https://semaphore.uqar.ca/id/eprint/2117/
