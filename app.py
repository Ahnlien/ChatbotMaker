from llama_index import SimpleDirectoryReader,GPTListIndex,GPTVectorStoreIndex,LLMPredictor,PromptHelper,ServiceContext,StorageContext,load_index_from_storage
from langchain import OpenAI
from flask import Flask, render_template, request, session, redirect, jsonify
import threading
import openai
import os
import sys
from model import answerMe, create_index, pdf_to_txt
import webbrowser

global pdf_directory
global txt_directory 
global vector_directory
global app_name

# Save the string to a text file
def save_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

# Load and read the text file
def load_text_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            text = file.read()
        return text
    else:
        return None

current_file_path = os.path.abspath(__file__)
# Get the parent directory path
parent_directory = os.path.dirname(current_file_path)

app_name = load_text_from_file(parent_directory+"/app_name.txt")
pdf_directory = parent_directory + "/pdfdata"
txt_directory = parent_directory + "/train_data"
vector_directory = parent_directory + "/vectors"


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    global app_name
    if(app_name == None or app_name == ""):
        if (request.method == "POST"):
            # Get the web app name from the form
            app_name = request.form.get("app_name")
            save_text_to_file(app_name,parent_directory+"/app_name.txt")
            print(vector_directory)
            
            # Perform any necessary validation on the directories
            # Get a list of PDF files in the directory
            pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

            # Convert each PDF file to TXT
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_directory, pdf_file)
                txt_file = os.path.splitext(pdf_file)[0] + '.txt'
                txt_path = os.path.join(txt_directory, txt_file)
                pdf_to_txt(pdf_path, txt_path)

            print("Conversion complete!")
            # update vectors
            try:
                create_index(txt_directory,vector_directory)
            except Exception as e:
                print("Error updating data:", e)

            # Redirect to the web app page
            return render_template("index.html",app_name=app_name)
        
        return render_template("setup.html")
    return render_template("index.html",app_name=app_name)

# Define the /api route to handle POST requests
@app.route("/api", methods=["POST"])
def api():
    # Get the message from the POST request
    message = request.json.get("message")
    # Send the message to OpenAI's API and receive the response
    response = answerMe(message,vector_directory)

    print(vector_directory)
    
    return {"content" : response}

@app.route("/update", methods=["GET"])
def update_index():

    # Directory containing the PDF files
    # pdf_directory = '/Users/ahn/Workspace/webapp/pdf_data'

    # Directory to save the TXT files
    # txt_directory = '/Users/ahn/Workspace/webapp/train_data'

    # Iterate over each PDF file in the directory
    pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith('.pdf')]

        # Convert each PDF file to TXT
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        txt_file = os.path.splitext(pdf_file)[0] + '.txt'
        txt_path = os.path.join(txt_directory, txt_file)
        pdf_to_txt(pdf_path, txt_path)

    print("Conversion complete!")
    # update vectors
    try:
        create_index(txt_directory,vector_directory)
        return jsonify({"success": True})
    except Exception as e:
        print("Error updating data:", e)
        return jsonify({"success": False})

if __name__=='__main__':
    webbrowser.open('http://localhost:8080/')
    app.run(host='0.0.0.0', port=8080, threaded=True)
    
