from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from chembl_webresource_client.new_client import new_client
# 1. Fetch ligand from ChEMBL
# =========================================================
def fetch_ligand(chembl_id):
    molecule = new_client.molecule.get(chembl_id)
    return molecule["molecule_structures"]["canonical_smiles"]

# =========================================================
# 2. Initial ligand (2D → 3D, no optimization)
# =========================================================
def initial_ligand(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol

# =========================================================
# 3. Prepared ligand (optimized)
# =========================================================
def prepared_ligand(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    return mol

# =========================================================
# 4. AI feature extraction
# =========================================================
def ligand_features(mol):
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2)
    }

# =========================================================
# 5. AI-based classification (explainable rules)
# =========================================================
def classify_ligand(features):
    if features["MW"] > 700 or features["RotB"] > 10:
        return "Problematic"
    elif features["MW"] < 300:
        return "Fragment-like"
    else:
        return "Drug-like"

# =========================================================
# 6. AI quality score (0–100) – ARTICLE-READY
# =========================================================
def ai_quality_score(features):
    score = 100

    if features["MW"] > 700:
        score -= 25
    if features["MW"] < 250:
        score -= 10
    if features["RotB"] > 10:
        score -= 20
    if features["LogP"] > 5:
        score -= 15
    if features["TPSA"] > 140:
        score -= 10

    return max(score, 0)

# =========================================================
# 7. Decision message
# =========================================================
def decision_message(lig_class):
    if lig_class == "Problematic":
        return "⚠️ Ligand may show unreliable docking behavior"
    elif lig_class == "Fragment-like":
        return "ℹ️ Ligand is small and suitable for fragment-based docking"
    else:
        return "✅ Ligand suitable for classical molecular docking"
# 9. MAIN PIPELINE FUNCTION (USED BY FLASK)
# =========================================================
def run_ligand_pipeline(chembl_id):

    smiles = fetch_ligand(chembl_id)

    mol_before = initial_ligand(smiles)
    mol_after = prepared_ligand(smiles)

    features = ligand_features(mol_after)
    lig_class = classify_ligand(features)
    ai_score = ai_quality_score(features)
    decision = decision_message(lig_class)

    return {
        "chembl_id": chembl_id,
        "features": features,
        "class": lig_class,
        "ai_score": ai_score,
        "decision": decision,
        "pdb_before": Chem.MolToPDBBlock(mol_before),
        "pdb_after": Chem.MolToPDBBlock(mol_after),
    }
