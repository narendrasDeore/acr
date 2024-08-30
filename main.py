import json
import os
from dateutil import parser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm_extraction = ChatOpenAI(model_name="gpt-4", temperature=0.0)

prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="""
    Extract the following information from the provided text and format it as a JSON dictionary:

    - First Name
    - Last Name
    - Gender
    - Date of Birth (DOB)
    - Height
    - Weight
    - Insurance
    - Policy Number
    - Medical Record Number
    - Hospital Record Number

    Example JSON format:
    {{
        "first name": "Patient First Name",
        "last name": "Patient Last Name",
        "gender": "Gender",
        "dob": "DOB",
        "height": Height,
        "weight": Weight,
        "insurance": "Insurance",
        "policy_number": "Policy Number",
        "medical_record_number": "Medical Record Number",
        "hospital_record_number": "Hospital Record Number"
    }}

    Text:
    {prompt}
    """
)

chain_extraction = LLMChain(llm=llm_extraction, prompt=prompt_template)

def extract_patient_info(prompt):
    extracted_info = chain_extraction.run(prompt=prompt)

    if not extracted_info:
        return {"error": "Model response is empty or invalid."}

    try:
        patient_data = json.loads(extracted_info)
        patient_data['dob'] = format_date(patient_data.get('dob', ''))
    except json.JSONDecodeError:
        return {"error": "Error decoding JSON from the model response."}

    return patient_data

def format_date(date_str):
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return date_str

def prompt_for_missing_info(patient_data):
    required_fields = [
        'first name', 'last name', 'gender', 'dob', 'height', 'weight',
        'insurance', 'policy_number', 'medical_record_number', 'hospital_record_number'
    ]

    missing_fields = [field for field in required_fields if not patient_data.get(field)]

    if missing_fields:
        return missing_fields
    return []

def interactive_prompt():
    user_input = input("Enter patient information:\n")
    patient_data = extract_patient_info(user_input)

    if "error" in patient_data:
        print(patient_data["error"])
        return None

    missing_fields = prompt_for_missing_info(patient_data)

    while missing_fields:
        print("Missing information required for:")
        for field in missing_fields:
            value = input(f"Please provide the {field.replace('_', ' ')}:\n")
            patient_data[field] = value
        missing_fields = prompt_for_missing_info(patient_data)

    print("Final Patient Data:")
    for key, value in patient_data.items():
        print(f"{key}: {value}")

    # Save patient data to a file
    with open('patient_data.json', 'w') as f:
        json.dump(patient_data, f)

    return patient_data

db_user = "root"
db_password = "anas123"
db_host = "localhost:3306"
db_name = "acr1"

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

llm_sql = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

toolkit = SQLDatabaseToolkit(llm=llm_sql, db=db)

agent_executor = create_sql_agent(
    llm=llm_sql,
    toolkit=toolkit,
    verbose=True
)

def insert_patient_data(natural_language_prompt: str):
    try:
        response = agent_executor.run(natural_language_prompt)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

def get_user_input():
    # Read patient data from the file
    with open('patient_data.json', 'r') as f:
        patient_data = json.load(f)

    # Format the prompt from patient data
    prompt = (
        f"Add a new patient with first name {patient_data.get('first name')}, "
        f"last name {patient_data.get('last name')}, "
        f"gender {patient_data.get('gender')}, "
        f"dob {patient_data.get('dob')}, "
        f"height {patient_data.get('height')} cm, "
        f"weight {patient_data.get('weight')} kg, "
        f"insurance {patient_data.get('insurance')}, "
        f"policy number {patient_data.get('policy_number')}, "
        f"medical record number {patient_data.get('medical_record_number')}, "
        f"hospital record number {patient_data.get('hospital_record_number')}"
    )
    return prompt

if __name__ == "__main__":
    patient_data_variable = interactive_prompt()

    if patient_data_variable:
        user_prompt = get_user_input()
        result = insert_patient_data(user_prompt)
        print(result)

