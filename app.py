import os
import spacy
import pickle
import random
import PyPDF2

from flask import Flask, render_template, request

train_data = pickle.load(open('./train_data.pkl','rb'))
nlp_model = spacy.load('nlp_model')

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'resume/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

        import PyPDF2
        pdf = PyPDF2.PdfFileReader(file)
        string = ""
        str_final = ""
        output = ""
        for i in range(1):
            string += pdf.getPage(i).extractText()

    
        with open("text.txt", "w", encoding = 'utf-8') as f:
            f.write(string)
            str_final = " ".join(string.split('\n'))
            #print(str_final)
    
        doc = nlp_model(str_final)
        for ent in doc.ents:
            result = f'{ent.label_.upper():{30}}- {ent.text}'
            print(result)
            output += result

    return output

if __name__ == "__main__":
    app.run(port=4555, debug=True)