"""Prompt templates for Visual CoT."""

# ============================================================================
# Object Selection Prompts
# ============================================================================

OBJECT_SELECTION_SYSTEM_PROMPT = (
    "Let's play a game. I have an image and a complex question about it, "
    "and you will give me the name of object in the image you want to look at the most. "
    "Please follow the format of the following examples and give me an object name directly.\n"
)

OBJECT_SELECTION_COMPLETION_PROMPT = "Please select the object most related to the question.\n===\n"


# Few-shot example templates
OBJECT_SELECTION_EXAMPLE_CHAT = """Question: {question}
===
The most related option is {object_name}.

===
"""

OBJECT_SELECTION_EXAMPLE_COMPLETION = """Question: {question}
===
Options:
The most related option is {object_name}.

===
"""

OBJECT_SELECTION_QUERY_CHAT = """Question: {question}
===
Options: {object_list}
The most related option is"""

OBJECT_SELECTION_QUERY_COMPLETION = """Question: {question}
===
Options:
The most related option is"""


# ============================================================================
# Question Answering Prompts
# ============================================================================

QA_SYSTEM_PROMPT_CHAT = (
    "Let's play a game. I have an image and a complex question about it. "
    "I will provide you some information about the image in the context, "
    "and you will give me the possible answer and reason to the question. "
    "You must provide an answer and can not say unclear or unknown. "
    "Please follow the format and answer style of the following examples and complete the last example.\n"
)

QA_COMPLETION_PROMPT = "Please answer the question according to the above context.\n===\n"

QA_ANSWER_PREFIX_CHAT = (
    "Based on the given information, I must guess the most possible answer. Answer:\n"
)
QA_ANSWER_PREFIX_COMPLETION = "Answer: The answer is"


# Few-shot example templates
QA_EXAMPLE_WITH_COT = """Context: {context}
===
Question: {question}{choices}
{answer_prefix} {answer}. {rationale}

===
"""

QA_EXAMPLE_WITHOUT_COT = """Context: {context}
===
Question: {question}{choices}
{answer_prefix} {answer}

===
"""

QA_QUERY_WITH_COT = """Context: {context}
{scene_graph}
===
Question: {question}{choices}
{answer_prefix}"""

QA_QUERY_WITHOUT_COT = """Context: {context}
{scene_graph}
===
Question: {question}{choices}
{answer_prefix}"""


# ============================================================================
# BLIP2 Follow-up Question Prompts
# ============================================================================

BLIP2_FOLLOWUP_USER_PROMPT = (
    "You will to look at the {object_name} in the picture and find {observation}. "
    "To find the answer to {main_question}, you can ask one question about the {object_name}. "
    "Please tell me the question you want to ask directly."
)

BLIP2_FOLLOWUP_COMPLETION_PROMPT = (
    "I look at the {object_name} in the picture and find {observation}. "
    "To find the answer to {main_question}, I ask one question about the {object_name}. "
    "My question is:"
)

# ============================================================================
# BLIP2 Caption Prompts
# ============================================================================

BLIP2_GLOBAL_CAPTION_PROMPT = "An image of "

BLIP2_GLOBAL_CAPTION_QUESTION_PROMPT = (
    "Question: Please look at the picture and answer the following question. {question} Answer:"
)

BLIP2_LOCAL_CAPTION_PROMPT = (
    "Question: Look at the {object_name} in this image. "
    "Please give a detailed description of the {object_name} in this image. Answer:"
)

BLIP2_LOCAL_CAPTION_QUESTION_PROMPT = (
    "Question: Please look at the {object_name} and answer the following question. "
    "{question} Answer:"
)

BLIP2_VERIFY_THOUGHT_PROMPT = (
    "Question: Does this sentence match the facts in the picture? "
    "Please answer yes or no. Sentence: In this picture, {thought} Answer:"
)

BLIP2_CORRECT_THOUGHT_PROMPT = (
    "Question: Please correct the following sentence according to the image. Sentence: {thought}"
)

BLIP2_DETECT_OBJECT_FIRST_PROMPT = (
    "Give me the name of one object, creature, or entity in the image."
)

BLIP2_DETECT_OBJECT_BESIDES_PROMPT = (
    "Give me the name of one object, creature, or entity in the image besides {existing_objects}?"
)
