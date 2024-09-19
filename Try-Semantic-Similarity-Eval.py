# This code demonstrates how to evaluate the semantic similarity between two texts using the SemanticSimilarityEvaluator.

from llama_index.core.evaluation import SemanticSimilarityEvaluator

# Create an instance of the SemanticSimilarityEvaluator
evaluator = SemanticSimilarityEvaluator()

# The following function is used to evaluate the similarity between two texts
# It uses the SemanticSimilarityEvaluator to evaluate the similarity between the response and the reference text
def evaluate_similarity():
    response = """Yoga offers a wide range of benefits for both the body and mind. 
    It can improve your flexibility, strength, and balance, while also reducing stress, anxiety, and pain. 
    Yoga also promotes better sleep, heart health, and overall well-being.  
    Whether you're looking for a physical workout or a way to de-stress, yoga can be a great option."""
    
    reference = """Yoga offers a wide range of benefits for both the body and mind. 
    It can improve your flexibility, strength, and balance, while also reducing stress, anxiety, and pain. 
    Yoga also promotes better sleep, heart health, and overall well-being.  
    Whether you're looking for a physical workout or a way to de-stress, yoga can be a great option.
    """
    # The aevaluate function returns a dictionary with the score and passing key-value pairs
    result = evaluator.evaluate(
        response=response,
        reference=reference,
    )

    print("Score: ", result.score)
    print("Passing: ", result.passing)  # default similarity threshold is 0.8

# The following function is used to run the async function
def main(): # convert this to entry point of python script
    evaluate_similarity()

# write a entry point to this python script
if __name__ == "__main__":
    main()



