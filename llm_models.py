from langchain_ollama import ChatOllama

messages = [('system', 'You are a helpful assistant'),
            ('human','what are you, respond with NAME: {your model name} \n\n Then, explain')]


print("_________________________________________________")
print("_________________________________________________")
# 8b model
# run ollama pull llama3.1 before executing this 
attacker_agent = ChatOllama(model = 'llama3.1')
# response = attacker_agent.invoke(messages)
# print(response.content)

# print("_________________________________________________")
# print("_________________________________________________")
# 1.5b model
# run ollama pull deepseek-r1:1.5b before executing this
planner_agent = ChatOllama(model = 'deepseek-r1:1.5b')

# response = planner_agent.invoke(messages)
# print(response.content)


# print("_________________________________________________")
# print("_________________________________________________")
# 2b model
# run ollama pull qwen3.5:2b 
evaluator_agent = ChatOllama(model = 'qwen3.5:2b')
# response = evaluator_agent.invoke(messages)
# print(response.content)
# print("_________________________________________________")
# print("_________________________________________________")

target_agent = ChatOllama(model = 'qwen3.5:0.8b')
# response = target_agent.invoke(messages)
# print(response.content)
# print("_________________________________________________")
# print("_________________________________________________")