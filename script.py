from dotenv import load_dotenv
import os

from imap_tools import MailBox, AND
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup

def get_informations_env():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    user_gmail = os.getenv("user")
    password_gmail = os.getenv("password")
    return groq_key, user_gmail, password_gmail


def get_messages(user_gmail, password_gmail, api_key):
    with MailBox("imap.gmail.com").login(user_gmail, password_gmail) as mb:
        for msg in mb.fetch(mark_seen=False):
            if msg.html:
                soup = BeautifulSoup(msg.html, "html.parser")
                email_text = soup.get_text(separator="\n", strip=True)
                _reading_gmails(email_text, api_key)


def _personality_model(api_key):
    system_prompt = "You are a email parser from pagbank, your objective is collect informations of value from email content sent they."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human)])
    chat = ChatGroq(temperature=0, model="llama-3.3-70b-versatile", groq_api_key=api_key)
    parser = StrOutputParser()
    return prompt, chat, parser

def _reading_gmails(email_content, api_key):
    prompt, chat, parser = _personality_model(api_key)
    chain = prompt | chat | parser
    response = chain.invoke({"text": f"Email content:{email_content}"})
    print(response)

if __name__ == "__main__":
    groq_key, user_gmail, password_gmail = get_informations_env()
    get_messages(user_gmail, password_gmail, groq_key)
