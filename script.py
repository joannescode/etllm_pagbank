from dotenv import load_dotenv
import os

from imap_tools import MailBox, AND
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
import re
import pandas as pd


def get_informations_env():
    """
    Loads environment variables from a .env file and retrieves specific values.

    This function uses the `load_dotenv` function to load environment variables
    from a .env file located in the current working directory. It then retrieves
    the values of the following environment variables:
    - GROQ_API_KEY: The API key for the GROQ service.
    - user: The Gmail username.
    - password: The Gmail password.

    Returns:
        tuple: A tuple containing the GROQ API key, Gmail username, and Gmail password.
    """
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    user_gmail = os.getenv("user")
    password_gmail = os.getenv("password")
    return groq_key, user_gmail, password_gmail


def get_messages(user_gmail, password_gmail, api_key):
    """
    Fetches and processes emails from a Gmail account.

    Args:
        user_gmail (str): The Gmail address to log in.
        password_gmail (str): The password for the Gmail account.
        api_key (str): The API key used for processing email content.

    Returns:
        list: A list of processed information extracted from the emails.

    """
    with MailBox("imap.gmail.com").login(
        user_gmail, password_gmail, initial_folder="[Gmail]/All Mail"
    ) as mb:
        all_informations = []
        for msg in mb.fetch(AND(from_="comunicacao@pagbank.com.br"), mark_seen=False):
            if msg.html:
                soup = BeautifulSoup(msg.html, "html.parser")
                email_text = soup.get_text(separator="\n", strip=True)
                response = _reading_gmails(email_text, api_key)
                all_informations.append(response)
                return all_informations


def _personality_model(api_key):
    """Initializes and returns the components required for an advanced email parser specialized in extracting financial transaction data from emails.
    Args:
        api_key (str): The API key required to access the Groq API.
    Returns:
        tuple: A tuple containing the following elements:
            - prompt (ChatPromptTemplate): The prompt template configured with the system and human messages.
            - chat (ChatGroq): The chat model configured with the specified temperature and model.
            - parser (StrOutputParser): The parser used to extract the required information from the chat response.
    The system prompt instructs the model to extract the following information from the email content:"""

    system_prompt = """You are an advanced email parser specialized in extracting financial transaction data from emails. 
    Your goal is to accurately extract key payment details. 
    
    Extract the following information from the email content:
    - Pagador: The name of the person who made the payment.
    - Banco Pagador: The name of the bank responsible for the payment.
    - Total Líquido: The total amount received.
    
    Just return informations needed from bullet point, not more.
    """

    human = "{text}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human)]
    )
    chat = ChatGroq(
        temperature=0, model="llama-3.3-70b-versatile", groq_api_key=api_key
    )
    parser = StrOutputParser()
    return prompt, chat, parser


def _reading_gmails(email_content, api_key):
    """
    Processes the content of an email using a personality model.

    Args:
        email_content (str): The content of the email to be processed.
        api_key (str): The API key required to access the personality model.

    Returns:
        dict: The result of invoking the personality model chain with the provided email content.
    """
    prompt, chat, parser = _personality_model(api_key)
    chain = prompt | chat | parser
    return chain.invoke({"text": f"Email content:{email_content}"})


def transform_messages_of_llm(response):
    """
    Extracts buyer names, bank information, and payment amounts from a given response string.

    Args:
        response (str): The response string containing the information to be extracted.

    Returns:
        tuple: A tuple containing three lists:
            - buyers (list of str): A list of buyer names extracted from the response.
            - banks (list of str): A list of bank information extracted from the response.
            - amounts (list of str): A list of payment amounts extracted from the response, formatted as strings.
    """
    buyer_pattern = r"(?:Informação do pagador:|Pagador:)\s*([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)+)"
    bank_pattern = r"(?:Banco Pagador:|Informação sobre o banco:)\s*([^\n]+)"
    amount_pattern = r"(?:Total Líquido:|Informação do pagamento:)\s*(?:\n\s*)?R\$ ?(\d{1,3}(?:\.\d{3})*,\d{2})"

    buyers = re.findall(buyer_pattern, response)
    banks = re.findall(bank_pattern, response)
    amounts = re.findall(amount_pattern, response)
    return buyers, banks, amounts


def collect_informations(buyers, banks, amounts):
    """
    Collects and organizes information from buyers, banks, and amounts into separate lists.

    Args:
        buyers (list): A list of buyer names.
        banks (list): A list of bank names.
        amounts (list): A list of amounts.

    Returns:
        tuple: A tuple containing three lists:
            - buyer_list (list): A list of buyer names, with "Não encontrado" for missing entries.
            - bank_list (list): A list of bank names, stripped of whitespace, with "Não encontrado" for missing entries.
            - amount_list (list): A list of amounts formatted as "R$ {amount}", with "Não encontrado" for missing entries.
    """
    buyer_list = []
    bank_list = []
    amount_list = []

    total = max(len(buyers), len(banks), len(amounts))
    for i in range(total):
        buyer = buyers[i] if i < len(buyers) else "Não encontrado"
        bank = banks[i].strip() if i < len(banks) else "Não encontrado"
        amount = f"R$ {amounts[i]}" if i < len(amounts) else "Não encontrado"

        buyer_list.append(buyer)
        bank_list.append(bank)
        amount_list.append(amount)
    return buyer_list, bank_list, amount_list


def convert_to_dataframe(buyer_list, bank_list, amount_list):
    """
    Converts lists of buyers, banks, and amounts into a pandas DataFrame.

    Args:
        buyer_list (list): A list of buyers.
        bank_list (list): A list of banks.
        amount_list (list): A list of amounts.

    Returns:
        pandas.DataFrame: A DataFrame containing the provided data with columns 'COMPRADOR', 'BANCO', and 'VALOR'.
    """
    dict = {"COMPRADOR": buyer_list, "BANCO": bank_list, "VALOR": amount_list}
    df = pd.DataFrame(dict)
    return df