
llama3 = {
    "config_list": [
        {
            "model": "LMstudio/llama3-zh",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

phi3 = {
    "config_list": [
        {
            "model": "LMstudio/llama3-zh",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}




from autogen import ConversableAgent

# jack = ConversableAgent(
#     "Jack (Phi-3)",
#     llm_config=phi3,
#     #system_message="Your name is Jack and you are a comedian in a two-person comedy show.",
#     system_message="你的名字叫Jack，你是一个中文AI作家。你的角色是根据指定主题创作引人入胜且信息丰富的文章，并且根据你的同事Emma的建议来修改和完善你创作的文章，每当你收到Emma的建议时，都要根据Emma的建议给出修改和完善后的完整文章。",
# )
# emma = ConversableAgent(
#     "Emma (llama3)",
#     llm_config=llama3,
#     #system_message="Your name is Emma and you are a comedian in two-person comedy show.",
#     system_message="你的名字叫Emma，你的角色是一个中文AI文章评审员。你的任务是针对你的同事Jack所写的文章评估并提出改进建议，每次对话你都要对文章作出评估并给出修改建议。",

# )

# chat_result = emma.initiate_chat(jack, message="Jack，请用中文写一篇关于男同的文章。", max_turns=2)

jack = ConversableAgent(
    "Jack",
    llm_config=llama3,
    #system_message="Your name is Jack and you are a comedian in a two-person comedy show.",
    system_message="Your name is Jack, and you are an expert in prompt engineering with years of experience optimizing various AI model prompts to improve and enhance prompts to improve AI model performance. You will optimize original_prompt based on the user's task_type and uesr_requirements, and tell Emma the optimization result.",
)
emma = ConversableAgent(
    "Emma",
    llm_config=llama3,
    #system_message="Your name is Emma and you are a comedian in two-person comedy show.",
    system_message="Your name is Emma, and you're an experienced AI QA expert with a focus on evaluating the effectiveness of prompts, a keen eye for detail, and a deep understanding of user needs. You will analyze Jack's optimization results and suggest your improvements, which will be analyzed and suggested for each round of dialogue.",
)

chat_result = emma.initiate_chat(jack, message=f'''Jack,please help me optimize the following prompt:
            Please write an article about gay men.The task_type is writing and the user_requirements is 200 words or less. 
            Please output the improved results.''',
            max_turns=2)