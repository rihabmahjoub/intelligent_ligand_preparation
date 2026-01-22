from flask import Flask, render_template, request
from ligand_ai_pipeline import run_ligand_pipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    if request.method == "POST":
        chembl_id = request.form.get("chembl_id")
        data = run_ligand_pipeline(chembl_id)

    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)

