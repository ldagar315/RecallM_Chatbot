# Blog: Building a Retrieval-Augmented Generation (RAG) Chatbot Without Vector Databases

Retrieval-Augmented Generation (RAG) is a promising technique that combines retrieval and generation to provide detailed and contextualized answers based on one's personal database or knowledge source. Initially RAG systems relied only on vector databases to search and retrieve documents using semantic embeddings, but there has been continuous research to find alternate ways storing and retrieving information. 

In this blog, we will demonstrate how to build a RAG chatbot without relying on vector databases, instead using Neo4J graph databases for retrieval, extracting concepts from research papers fetched in realtime from Arxiv. The chatbot is based on the concepts from the paper titled ***[RecallM: An Adaptable Memory Mechanism with Temporal Understanding for Large Language Models.](https://arxiv.org/pdf/2307.02738)***
The chatbot not only retrieves relevant papers based on user query but also generates meaningful responses based on previously stored knowledge, ensuring no unnecessary re-fetching of previously encountered information.

----------

### 1. The Problem with Vector Databases

Vector databases typically store high-dimensional embeddings of textual data, which is a very convienient way of understanding the semantic meaning and making relations. However, this method has its drawbacks:

-   **Memory Overhead**: Storing and querying embeddings is resource-intensive and creating embeddings for a large database takes a lot of time, so it is not scalable. 
    
-   **Lack of Temporal Knowledge**: Vector databases treat data as static and don't easily support updates for evolving information.
    
-   **Belief Updates**: They struggle with incorporating new knowledge in real-time, particularly when context needs to be updated dynamically.

----------

### 2. RAG Without Vector Databases

#### RecallM: A Novel Approach to Memory in LLMs

This project draws inspiration from concepts introduced in the RecallM architecture, a novel memory mechanism designed to overcome limitations in traditional retrieval methods. 
The RecallM research paper proposes a graph-based approach to storing long-term memory for Large Language Models (LLMs), where temporal knowledge and belief updating are core challenges. Rather than relying on vector databases, RecallM utilizes a neuro-symbolic architecture to build a memory mechanism that can dynamically update and retain temporal relationships between concepts. This makes it four times more effective than vector databases for long-term memory tasks like belief updating (as claimed in the text). 

The chatbot uses Neo4J, a graph-based database, instead of a vector database. This allows us to:

1.  Store relationships between concepts (entities/nouns) and papers in a structured way.
    
2.  Retrieve relevant information based on the user’s question by querying the graph for connected nodes.
    
3.  Update knowledge dynamically by adding new concepts only when they are encountered for the first time.
    

The chatbot fetches papers from Arxiv, extracts relevant concepts (proper and common nouns as of now), and stores them in a graph database. It only fetches new papers when the user asks a question involving concepts not yet in the database.

----------
### 3. Architecture Overview

The architecture is mainly divided into 2 parts:
1. Storage
2. Retrieval 

#### Storage
![](https://ar5iv.labs.arxiv.org/html/2307.02738/assets/images/KU_pipeline.png)
This is the architecture of the information storage part of the graph, where firstly all of the nouns (i.e. concepts or important words) from the  whole text corpus (i.e. documents) are identified using Part of speech tagger. 
Secondly the indentified nouns are stemmed (i.e. run, ran, running all converted to run) for the all the nous/concepts and merged into one. 
After that the relevant contexts (sentences) where the nouns/concepts were used are identified and stored as child graph nodes for the concepts/nouns in a chronological manner that they appear in the document along with the time stamp when they stored. 
For updating the database whenever a new context is added the existing concept related to that is fetched and new context is appended at the end of it with an updated time stamp.
For new concepts an entire new node is created. 

#### Retrieval
When querying
![enter image description here](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYLn2yozFWW_SLRc-jnT__26eLMrA1dRRzRA&s)

1. Concepts (i.e. nouns) are extracted from user question
2. The concepts are then matched with the existing concepts in the graph
3. All of the neighboring concept nodes of the matched nodes are returned. 
4. All of the contexts related to the all of the concepts are returned and parsed through a LLM to formulate a response. 

----------


### 4. The Full Project: Code Walkthrough

Now, let’s dive into the code and explore how we built this system from scratch. Before diving into the code lets see an overall picture about what we would be doing. 

- First we would initializing an online graph database called Neo4j. 
- Then we define a function to fetch papers from Arxiv.  
 - After which a step by step implementation of all of the functions (first storage,    then retrieval related) is defined.  
 - And    at last      all of the function are put together to make a chatbot    using Gemini.

#### Importing Libraries and Initializing Graph from Neo4j

    import nltk
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    import urllib.request as libreq
    import xmltodict
    
    # go to Neo4j website and sign up for a free instance  
    from py2neo import Graph
    username = 'your_username'
    password = 'your_password'
    url = 'the_url_of_the_graph'
    graph = = Graph(url, auth=(username, password))

  

#### Fetching Papers from Arxiv

    def fetch_papers(query, max_results=5):
	    with libreq.urlopen(f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}') as url:
		    r = url.read()
		    xml_data = xmltodict.parse(r)
		papers = []
		for result in xml_data['feed']['entry']:
			papers.append({
				'title' : result['title'],
				'summary': result['summary'],
				'published': result['published'] })
		return papers 


This function fetches papers from Arxiv based on a user query. We limit the results to a configurable max_results (default is 5). The Arxiv api currently only returns response in XML format so we use python library xmltodict to parse the result into a python dictionary.
We are currently using only 3 attributes from the API result for our chatbot, i.e. the paper title, published date and a summary provided, for a better performing chatbot one can also download the relevant texts from the arxiv and then parse and store the whole paper in memory. 

#### Extracting Concepts from Papers

    def extract_concepts(text):
	    tokens = word_tokenize(text)
	    tagged = pos_tag(tokens)
	    concepts = [word for word, pos in tagged if pos in ['NN', 'NNP']]
	    return concepts

  

Once the papers are fetched, we extract the nouns (concepts) from the titles and summaries of the papers. These will serve as the nodes in our graph database. We use the POS tagging capability from the nltk library to identify nouns.

#### Creating Stemming from the words
    from nltk.stem import PorterStemmer
    def  stem_concepts(concepts):
    
	    # Initialize the Porter Stemmer
	    stemmer = PorterStemmer()
	    
	    # Apply stemming to each concept in the list
	    stemmed_concepts = [stemmer.stem(concept) for concept in concepts]  
	     
	    return stemmed_concepts

To tackle concept duplicacy, we convert each concept into a stemming of the word i.e. running, ran, run all are converted to the stemming run. 

#### Extracting Contexts

    def  extract_contexts(stemmed_concepts, text):
    
	    # Split the text into sentences
	    sentences = nltk.sent_tokenize(text)
	    
	    # Dictionary to store concepts and their corresponding context (sentences)
	    concept_context = {concept: [] for concept in stemmed_concepts}
	    
	    # Iterate over sentences and match concepts within each sentence
	    for sentence in sentences:
		    words = nltk.word_tokenize(sentence)
	    for concept in stemmed_concepts:
		    if concept in words: # Check if the concept appears in the sentence
			    concept_context[concept].append(sentence) # Add sentence as context
	    return concept_context

Once we have all of the stemmings we then extract the context (i.e. sentences) where they are used. This is done by first breaking down the entire document into sentences and then associating concepts (nouns) with the contexts (sentences).

#### Storing Papers in the Graph

   

     def  store_in_graph(paper):
    
	    # Store the paper details as a node
	    graph.run("""
	    MERGE (p:Paper {title: $title, published: $published, summary: $summary})
	    """, title=paper['title'], published=str(paper['published']), summary=paper['summary'])
	    
	    # Iterate over the concepts and store them with context in the graph
	    for concept, contexts in paper['concepts'].items():
	    
		    # Store the concept node
		    graph.run("MERGE (c:Concept {name: $name})", name=concept)
		    
		    # Link the paper to the concept
		    graph.run("""
		    MATCH (p:Paper {title: $title}), (c:Concept {name: $name})
		    MERGE (p)-[:HAS_CONCEPT]->(c)
		    """, title=paper['title'], name=concept)
		    
		    # For each context (sentence) related to the concept, store it and link it in chronological order
		    for idx, context in  enumerate(contexts):
			    graph.run("""
			    MERGE (ctx:Context {sentence: $sentence, order: $order})
			    """, sentence=context, order=idx)
			    
			    # Link the concept to its context in the correct order
			    graph.run("""
			    MATCH (c:Concept {name: $name}), (ctx:Context {sentence: $sentence})
			    MERGE (c)-[:HAS_CONTEXT {order: $order}]->(ctx)
			    """, name=concept, sentence=context, order=idx)

	  

We use Neo4J to store the papers and their concepts. The papers are represented as nodes, and each concept is connected to the paper using the HAS_CONCEPT relationship. This graph structure allows us to easily query the database for papers relevant to a user's question.

#### Merging and updating nodes

    def  merge_and_update_nodes(concept, context, current_time):
    
	    # Merge the concept node if it exists, or create a new one
	    graph.run("""
	    MERGE (c:Concept {name: $name})
	    ON CREATE SET c.created_at = $current_time, c.updated_at = $current_time
	    ON MATCH SET c.updated_at = $current_time
	    """, name=concept, current_time=current_time)
	    
	    # Merge the context (sentence) and maintain the chronological order
	    graph.run("""
	    MERGE (ctx:Context {sentence: $sentence})
	    ON CREATE SET ctx.created_at = $current_time, ctx.order = size((c)-[:HAS_CONTEXT]->()) + 1
	    """, sentence=context, current_time=current_time)
	    
	    # Link the concept to the new context in chronological order
	    graph.run("""
	    MATCH (c:Concept {name: $name}), (ctx:Context {sentence: $sentence})
	    MERGE (c)-[:HAS_CONTEXT {order: ctx.order, updated_at: $current_time}]->(ctx)
	    """, name=concept, sentence=context, current_time=current_time)
	    
	    # Update temporal information by setting 'updated_at' and linking in proper sequence
	    graph.run("""
	    MATCH (c:Concept {name: $name})
	    SET c.updated_at = $current_time
	    """, name=concept, current_time=current_time)
This function updates the context in the existing node if the concept exists or creates a new concept, and then stores the contexts with the temporal information (time) when the node was last updated. 

#### Updating Temporal Index and Relationship Strength

    def update_temporal_and_strength(concept, related_concepts, current_time):
        
        # Update temporal index for the concept itself
        graph.run("""
        MATCH (c:Concept {name: $name})
        SET c.updated_at = $current_time
        """, name=concept, current_time=current_time)
	    
	    # For each related concept, update the relation strength and temporal index
	    for related_concept in related_concepts:
	        # Merge the related concept if it doesn't exist
	        graph.run("""
	        MERGE (r:Concept {name: $related_name})
	        ON CREATE SET r.created_at = $current_time, r.updated_at = $current_time
	        ON MATCH SET r.updated_at = $current_time
	        """, related_name=related_concept, current_time=current_time)
	        
	        # Update the strength of the relationship between the concepts
	        graph.run("""
	        MATCH (c:Concept {name: $name})-[rel:RELATED_TO]-(r:Concept {name: $related_name})
	        MERGE (c)-[rel:RELATED_TO]->(r)
	        ON CREATE SET rel.strength = 1, rel.updated_at = $current_time
	        ON MATCH SET rel.strength = rel.strength + 1, rel.updated_at = $current_time
	        """, name=concept, related_name=related_concept, current_time=current_time)

This function ensures that both **temporal relevance** and **relationship strength** between concepts are properly managed when new information is added. 
The **strength** of the relationship between the concept and each related concept is updated. If the relationship is new, its strength is initialized to 1. If it already exists, the strength is incremented by 1, simulating a learning process where the connection strengthens with repeated mentions.
 The `updated_at` timestamp for the **concept** and all its **related concepts** is set to the current time, indicating when the concept or relationship was last updated.
For each related concept, the function checks if it already exists, and if not, it creates it with the proper `created_at` and `updated_at` timestamps.

#### Querying the graph for relevant responses

    def query_relevant_concepts(essential_concepts, temporal_window):
        # Create a list to store relevant concepts and their relations
        relevant_concepts = []
	    
	    # Loop through each essential concept and query the graph for related concepts
        for concept in essential_concepts:
            # Query to get related concepts within the temporal window
            query = """
            MATCH (c:Concept {name: $name})-[rel:RELATED_TO]-(related:Concept)
            WHERE rel.updated_at >= $temporal_limit
            RETURN related.name AS related_name, rel.strength AS strength, rel.updated_at AS updated_at
            ORDER BY rel.updated_at DESC, rel.strength DESC
            """
            # Execute the query with the given temporal window limit
            result = graph.run(query, name=concept, temporal_limit=temporal_window)
            
            # Append results to the relevant_concepts list
            for record in result:
                relevant_concepts.append({
                    'name': record['related_name'],
                    'strength': record['strength'],
                    'updated_at': record['updated_at']
                })
        
        return relevant_concepts
This function focuses on querying the graph database and considering temporal constraints, ensuring only relevant, recent connections are returned.

#### Sorting the retrieved queries

    def sort_concepts_by_temporal_and_strength(concepts):
        # Sort concepts by two keys: first by 'updated_at' (descending) and then by 'strength' (descending)
        sorted_concepts = sorted(concepts, key=lambda x: (x['updated_at'], x['strength']), reverse=True)
    
    return sorted_concepts
This function sorts the retrieved concepts based on their **temporal index** (most recent first) and **relation strength** (strongest first).

#### Extracting Concepts from the sorted concepts
    def extract_contexts_for_prompt(sorted_concepts):
        # Initialize a list to hold the contexts for the prompt
        contexts = []
    
        # Iterate over the sorted concepts and retrieve their contexts from the graph
        for concept in sorted_concepts:
            # Query to get contexts associated with each concept
            query = """
            MATCH (c:Concept {name: $name})-[:HAS_CONTEXT]->(ctx:Context)
            RETURN ctx.sentence AS context
            """
            # Execute the query
            result = graph.run(query, name=concept['name'])
    
            # Append the retrieved contexts to the list
            for record in result:
                contexts.append(record['context'])
    
        return contexts


#### Initializing the Gemini Model to Create Responses

    import google.generativeai as genai
    genai.configure(api_key= 'your-google-api-key')
    model = genai.GenerativeModel("gemini-1.5-flash")

  

For the generative part of the chatbot, we are using Google’s Gemini 1.5 Flash model to generate responses based on the papers returned by the graph. This model is faster and fits well for our use case of a chatbot. The generated text is based on the context provided by the relevant papers.

#### The Chatbot Logic


    def chatbot():
	    memory = set() # Store previously asked concepts
	    print("Welcome to the RAG Chatbot! Ask any question related to research papers.\nType exit to stop the chatbot")

        while True:
	        user_question = input("\nYou: ")
	        if user_question.lower() == "exit":
		        break
    
		    # Extract concepts from the user's question
		    extracted_concepts = set(extract_concepts(user_question))

		    # Check for new concepts
		    new_concepts = extracted_concepts - memory
    
		    if new_concepts:
			    memory.update(new_concepts)
		    papers = fetch_papers(" ".join(new_concepts), max_results=5)
		    for paper in papers:
			    paper['concepts'] = stem_concepts(extract_concepts(paper['title'] + ' ' + paper['summary']))
			    for concepts in paper['concepts']:
				    paper['concepts']['contexts'] = extract_context(concepts, paper)
				merge_and_update_nodes(paper['concepts'], paper['concepts']['contexts'], current_time)
				update_temporal_and_strength(paper['concepts'],new_concepts, current_time)
			    store_in_graph(paper)
    
		    # Query graph for relevant papers and extract contexts
		    concepts = query_relevant_concepts(user_question, temporal_window=2)
		    sorted_concepts = sort_concepts_by_temporal_and_strength(concepts)
		    contexts_from_query = extract_contexts_for_prompt(sorted_concepts)
		    
    
		    if not related_papers:
			    print("I couldn't find any relevant papers.")
			    continue
    
		    # Generate a response
		    context = f'From the given extracted contexts from the from research papers =  {extract_contexts_from_prompts}, answer the following question: {user_question}'
		    response = model.generate_content(context)
		    print(f"\nChatbot: {response.text}")
    
    # Run the chatbot
    chatbot()

  

This chatbot keeps track of previously queried concepts and only fetches new papers when new concepts are introduced. It queries the graph for relevant papers and generates contextual responses based on the paper content.

----------

### 5. Conclusion

In this project we see the power and flexibility of building a RAG system without vector databases. By using graph databases, we can efficiently store and retrieve knowledge, ensuring faster response times and reducing redundant queries. This chatbot is particularly useful in academic and research contexts where the underlying knowledge evolves over time and needs to be dynamically updated.

Feel free to modify this project for your use case and explore new ways to improve the knowledge retrieval process without relying on traditional vector-based methods.


