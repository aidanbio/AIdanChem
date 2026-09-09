"""Run AIdanFold's deployed flow-matching checkpoint on trypsin + benzamidine,
extract the ligand pose, build an RDKit mol, and validate it with PoseBusters.

This is a feasibility smoke test: AIdanFold's flow-matching structure head
was trained/validated protein-only; here we test whether it produces a
physically plausible ligand pose when fed a protein+ligand complex via
ESMFold2's native (but until now untested, for this head) co-folding input.
"""
import sys, os, json

# Run from the AIdanFold working tree root (the checkout that has models/ +
# checkpoints/, not the git submodule reference) with that root on sys.path.
AIDANFOLD_ROOT = os.environ.get("AIDANFOLD_ROOT", ".")
sys.path.insert(0, os.path.join(AIDANFOLD_ROOT, "src"))
sys.path.insert(0, AIDANFOLD_ROOT)
os.chdir(AIDANFOLD_ROOT)

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

OUT_DIR = os.environ.get("PROBE_OUT_DIR", "./ligand_probe_out")
os.makedirs(OUT_DIR, exist_ok=True)

TRYPSIN = ("IVGGYTCGANTVPYQVSLNSGSHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN")
LIGAND_SMILES = "NC(=[NH2+])c1ccccc1"
# Path to the deployed flow-matching checkpoint (see docs/ARCHITECTURE.md section 4
# for how to obtain/select one); not committed here since checkpoint files/paths
# are internal to the AIdanFold project.
CKPT = os.environ["AIDANFOLD_CKPT"]
MODEL_DIR = "models/esmfold2_fast_cutoff2025"


def decode_atom_name(chars) -> str:
    return "".join(chr(int(c) + 32) for c in chars if int(c) != 0).strip()


def main():
    from transformers.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, LigandInput, StructurePredictionInput
    from common.esmfold2adv.dataprep.cache_trunk_features import capture_trunk
    from common.esmfold2adv.flow.evaluation.evaluate_flow import build_flow_sampler

    print("[1] loading ESMFold2 (bf16)...", flush=True)
    hf = ESMFold2ExperimentalModel.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    try:
        hf.set_kernel_backend(None)
    except Exception as e:
        print("  set_kernel_backend skipped:", e)
    hf = hf.cuda().eval()

    print("[2] building protein+ligand input...", flush=True)
    inp = StructurePredictionInput(sequences=[
        ProteinInput(id=["A"], sequence=TRYPSIN),
        LigandInput(id=["B"], smiles=LIGAND_SMILES),
    ])
    builder = ESMFold2InputBuilder()
    with torch.no_grad():
        features, chain_infos = builder.prepare_input(inp, device="cuda")

    print("[3] running trunk forward (capture_trunk, num_loops=20)...", flush=True)
    trunk_feats = capture_trunk(hf, features, num_loops=20)
    print("    captured keys:", sorted(trunk_feats.keys()))

    print(f"[4] loading flow head from {CKPT} ...", flush=True)
    sampler, epoch = build_flow_sampler(hf, CKPT, torch.device("cuda"))
    print(f"    loaded epoch={epoch}, self_cond={sampler.config.self_conditioning}, "
          f"solver={sampler.config.ode_solver}")

    n_atoms = features["ref_pos"].shape[1]
    print(f"[5] sampling {n_atoms} atom coordinates (20-step ODE, deployment config)...", flush=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
        coords = sampler.sample(
            trunk_feats=trunk_feats,
            num_atoms=n_atoms,
            num_sampling_steps=20,
        )
    coords = coords.float()
    print("    coords shape:", tuple(coords.shape))
    torch.save(coords.cpu(), os.path.join(OUT_DIR, "coords.pt"))

    # ── locate the ligand chain/tokens ──
    ligand_chain = [ci for ci in chain_infos if ci.mol_type == 3][0]
    protein_chain = [ci for ci in chain_infos if ci.mol_type == 0][0]
    lig_atom_start = ligand_chain.tokens[0].atom_start
    lig_atom_count = sum(t.atom_count for t in ligand_chain.tokens)
    print(f"[6] ligand atoms: [{lig_atom_start}, {lig_atom_start + lig_atom_count})")

    ref_element = features["ref_element"][0].cpu()
    ref_names_raw = features["ref_atom_name_chars"][0].cpu()
    lig_coords = coords[0, lig_atom_start:lig_atom_start + lig_atom_count, :].float().cpu()

    names = []
    elements = []
    for i in range(lig_atom_start, lig_atom_start + lig_atom_count):
        names.append(decode_atom_name(ref_names_raw[i]))
        elements.append(int(ref_element[i]))
    print("    names:", names)
    print("    elements (Z):", elements)
    print("    ligand_bonds:", ligand_chain.ligand_bonds)

    # ── build RDKit mol: atoms + bonds from ligand_bonds (topology only), ──
    # ── predicted 3D coords as the single conformer.                     ──
    name_to_idx = {n: i for i, n in enumerate(names)}
    mol = Chem.RWMol()
    for z in elements:
        mol.AddAtom(Chem.Atom(z))
    for a, b in ligand_chain.ligand_bonds:
        mol.AddBond(name_to_idx[a], name_to_idx[b], Chem.BondType.SINGLE)

    # Ring perception + aromatic/amidine bond-order assignment.
    ri = mol.GetRingInfo()
    Chem.FastFindRings(mol)
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 6 and all(elements[i] == 6 for i in ring):
            for i in range(6):
                a, b = ring[i], ring[(i + 1) % 6]
                mol.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.AROMATIC)
                mol.GetAtomWithIdx(a).SetIsAromatic(True)
                mol.GetAtomWithIdx(b).SetIsAromatic(True)

    # amidine/amidinium: C bonded to two N -> one C=N double bond, charge +1 on that N.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        n_neighbors = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 7]
        if len(n_neighbors) == 2:
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), n_neighbors[0].GetIdx())
            bond.SetBondType(Chem.BondType.DOUBLE)
            n_neighbors[0].SetFormalCharge(1)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = lig_coords[i].tolist()
        conf.SetAtomPosition(i, (x, y, z))
    mol.AddConformer(conf)

    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
        print("    RDKit sanitize: OK")
    except Exception as e:
        print("    RDKit sanitize FAILED:", e)

    sdf_path = os.path.join(OUT_DIR, "ligand_pose.sdf")
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    print(f"[7] wrote ligand pose -> {sdf_path}")
    print(Chem.MolToSmiles(mol))

    # ── minimal generic PDB writer for the protein chain (any residue) ──
    pdb_path = os.path.join(OUT_DIR, "receptor.pdb")
    ref_names_all = features["ref_atom_name_chars"][0].cpu()
    prot_coords = coords[0, :protein_chain.tokens[-1].atom_start + protein_chain.tokens[-1].atom_count, :].float().cpu()
    lines = []
    serial = 1
    for tok in protein_chain.tokens:
        for ai in range(tok.atom_start, tok.atom_start + tok.atom_count):
            name = decode_atom_name(ref_names_all[ai])
            elem = {6: "C", 7: "N", 8: "O", 16: "S"}.get(int(ref_element[ai]), "C")
            x, y, z = prot_coords[ai].tolist()
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} {tok.residue_name:<3s} A{tok.residue_index+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
            )
            serial += 1
    with open(pdb_path, "w") as f:
        f.write("\n".join(lines) + "\nEND\n")
    print(f"[8] wrote receptor -> {pdb_path} ({len(protein_chain.tokens)} residues)")

    # ── PoseBusters ──
    print("[9] running PoseBusters...", flush=True)
    from posebusters import PoseBusters
    buster = PoseBusters(config="mol")  # ligand-only geometry checks (no redock ref needed)
    df = buster.bust([sdf_path], None, None)
    print(df.T.to_string())
    df.to_csv(os.path.join(OUT_DIR, "posebusters_mol.csv"))

    buster_dock = PoseBusters(config="dock")  # + protein-ligand checks (clashes, etc.)
    df2 = buster_dock.bust([sdf_path], None, pdb_path)
    print(df2.T.to_string())
    df2.to_csv(os.path.join(OUT_DIR, "posebusters_dock.csv"))


if __name__ == "__main__":
    main()
