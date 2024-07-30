import autogen

config_list = [
    {
        "model": "LMstudio/llama3-zh",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio"
    }
]

llm_config={
    "config_list": config_list,
    "temperature": 0
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="AI agent"
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"use_docker": False },
)
user_proxy.initiate_chat(
    assistant,
    message="Write a python function to calculate the square root of a number, and call it with the number 4."
)