from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def system_prompt():
    system_prompt = ( 
                f"You are intelligent chatbot designed to help users answer questions related to electric vehicle charging stations provided in the context or website. Your role is to answer user questions from the context accurately "
        
            "Instructions:\n"


            "1. **Answer based on context only**: If the information is present, answer using the relevant content retrieved from the system.\n"
            "2. **If you donot find the information in the context, dont answer.**"
            "{context} exactly like as the extracted documents."
            )
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    return prompt


def history_prompt():

        # Define contextualize prompt
        contextualize_q_system_prompt =     "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "You Must not use the context if you have the answer"
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        return contextualize_q_prompt