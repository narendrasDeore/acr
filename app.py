from flask import Flask, request, jsonify
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
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

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

    return missing_fields


def generate_sql_prompt(patient_data):
    return (
        f"Add a new patient with first name {patient_data.get('first name')}, "
        f"last name {patient_data.get('last name')}, "
        f"gender {patient_data.get('gender')}, "
        f"dob {patient_data.get('dob')}, "
        f"height {patient_data.get('height')}, "
        f"weight {patient_data.get('weight')}, "
        f"insurance {patient_data.get('insurance')}, "
        f"policy number {patient_data.get('policy_number')}, "
        f"medical record number {patient_data.get('medical_record_number')}, "
        f"hospital record number {patient_data.get('hospital_record_number')}"
    )


@app.route('/add_patient', methods=['POST'])
def add_patient():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No 'text' field in request"}), 400

    user_input = data['text']
    patient_data = extract_patient_info(user_input)

    if "error" in patient_data:
        return jsonify({"error": patient_data["error"]}), 400

    missing_fields = prompt_for_missing_info(patient_data)

    if missing_fields:
        return jsonify({"missing_fields": missing_fields, "patient_data": patient_data}), 200

    sql_prompt = generate_sql_prompt(patient_data)

    try:
        result = agent_executor.run(sql_prompt)
        return jsonify({"message": "Patient added successfully", "result": result}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=8000)
