# This code demonstrates how to use the OpenAI API to generate responses to a conversation.
# It uses the ChatMessage class to represent messages in the conversation with roles for the system and user.
# The Chat Assistant will generate responses to the messages using the OpenAI API.

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

messages = [
    ChatMessage(
        role="system", 
        content="""You are a Teacher of Physics in a junior high school with funky sense of humour. 
        You are teaching a class of 12-year-olds. You are about to start a lesson on the solar system."""
    ),
    ChatMessage(role="user", content="What are you going to teach them today?"),
]
#resp = OpenAI().chat(messages)
#print(resp)

messages = [
    ChatMessage(
        role="system", 
        content="""I am an HR Strategist of a large corporation. I am about to start a meeting with the CEO to discuss the company's talent management strategy.
        I need to prepare a presentation on the latest trends in talent management.
        I need to present the latest trends in talent management to the CEO."""
    ),
    ChatMessage(role="user", content="What are the HR Technology Trends that the Organization may consider exploring#?"),
]
#resp = OpenAI().chat(messages)
#print(resp)


messages = [
    ChatMessage(
        role="system", 
        content="""I am an HR Strategist of a large corporation. I am about to start a meeting with the CEO to discuss the company's talent management strategy.
        I need to prepare a presentation on the latest trends in talent management.
        I need to present the latest trends in talent management to the CEO."""
    ),
    ChatMessage(role="user", content="What are the HR Technology Trends that the Organization may consider exploring#?"),
]
#resp = OpenAI().chat(messages)
#print(resp)