'''
cgenff_to_openmm.py
Convert CGenFF ligand stream files (.str) into OpenMM force field XML files (with ParmEd) that can be
loaded together with OpenMM's charmm36_2024.xml + charmm36_2024/water.xml.

Usage (in an environment with ParmEd):
    python cgenff_to_openmm.py <toppar dir> <out dir> <ligand.str> <template name> [<ligand.str> <template name> ...]
e.g.
    python cgenff_to_openmm.py ~/lab/GroEL/raw/toppar_c36_feb26/toppar ~/lab/GroEL/openMM_inputs/cgenff_xml \
        tmp_plus/tmp_plus.str TMPH  tmp/tmp.str TMP0  nadph/nadph.str NDP4

<toppar dir> must hold top_all36_cgenff.rtf / par_all36_cgenff.prm of the CGenFF version the .str files were
generated for (v5.0 for the current CGenFF server), and toppar_water_ions.str.

Why the output is post-processed:
1. charmm36_2024.xml already defines CGenFF atom classes (CG2R61, OG2P1, ...) from an older CGenFF, and OpenMM
   matches bonded, Urey-Bradley and Lennard-Jones parameters by class. With the plain CGenFF class names, a
   ligand would also pick up those parameters (Urey-Bradley terms get applied twice, +15.6 kcal/mol for NADPH),
   and two ligand files loaded together double-count each other's Urey-Bradley terms. So every atom type and
   class in a ligand file is prefixed with the template name (e.g. NDP4_OG2P1).
2. Prefixed classes no longer match CHARMM's pair-specific LJ corrections (NBFIX) for ions, e.g. Na+ with the
   phosphate oxygen OG2P1. Those are re-added from toppar_water_ions.str for the prefixed classes. NBFIX pairs
   whose partner isn't defined in the file (CGenFF halogens etc.) are dropped.
Template names must not clash with residues in charmm36_2024.xml (TMP, TMPP and NDP are taken).
'''
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import parmed

toppar, outdir, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(outdir, exist_ok=True)
base = [os.path.join(toppar, "top_all36_cgenff.rtf"), os.path.join(toppar, "par_all36_cgenff.prm")]
ion_stream = os.path.join(toppar, "toppar_water_ions.str")


def read_nbfix(path):
    '''(type1, type2) -> (Emin kcal/mol, Rmin A) for every NBFIX line in a CHARMM parameter/stream file'''
    out, on = {}, False
    for line in open(path, errors="ignore"):
        w = line.split("!")[0].split()
        if not w:
            continue
        key = w[0].upper()
        if key.startswith("NBFIX"):
            on = True
            continue
        if on and (key in ("END", "HBOND", "BONDS", "ANGLES", "DIHEDRALS", "IMPROPER", "IMPROPERS", "NONBONDED",
                           "CMAP", "RETURN") or key.startswith("READ")):
            on = False
        if on and len(w) >= 4:
            out[(w[0], w[1])] = (float(w[2]), float(w[3]))
    return out


ion_nbfix = read_nbfix(ion_stream)

for stream, template_name in zip(pairs[0::2], pairs[1::2]):
    # 1. ParmEd conversion: CGenFF base files + this ligand's .str, keeping only this ligand's residue template
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # CGenFF base files have harmless duplicate-parameter warnings
        params = parmed.charmm.CharmmParameterSet(*base, stream)
    resname = next(l.split()[1] for l in open(stream) if l.startswith("RESI"))
    res = params.residues[resname]
    res.name = template_name                         # template name only; topology residue names are set separately
    params.residues = {template_name: res}
    params.patches = {}                              # CGenFF patches (PRES) don't apply to these ligands
    lig_types = {a.type for a in res.atoms}
    tmp_out = os.path.join(outdir, f".{template_name}.raw.xml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parmed.openmm.OpenMMParameterSet.from_parameterset(params).write(
            tmp_out, write_unused=False, separate_ljforce=True, charmm_imp=True,
            provenance={"Source": [os.path.basename(f) for f in base + [stream]]})

    # 2. Prefix every atom type and class with the template name
    tree = ET.parse(tmp_out)
    root = tree.getroot()
    defined = {t.get("class") for t in root.iter("Type")}
    rename = {c: f"{template_name}_{c}" for c in defined}
    for el in root.iter():
        for attr in ("name", "class", "type", "class1", "class2", "class3", "class4",
                     "type1", "type2", "type3", "type4"):
            v = el.get(attr)
            if v and v in rename and not (el.tag in ("Residue", "Atom") and attr == "name"):
                el.set(attr, rename[v])

    # 3. NBFIX: keep only pairs within this file, then add ion pairs (e.g. SOD-OG2P1) for the prefixed classes
    lj = root.find("LennardJonesForce")
    for nb in list(lj.findall("NBFixPair")):
        if not (nb.get("class1") in rename.values() and nb.get("class2") in rename.values()):
            lj.remove(nb)
    added = []
    for (t1, t2), (emin, rmin) in ion_nbfix.items():
        for lig, other in ((t1, t2), (t2, t1)):
            if lig in lig_types and other not in lig_types:
                ET.SubElement(lj, "NBFixPair", {"class1": other, "class2": rename[lig],
                                                "sigma": repr(rmin / 2 ** (1 / 6) / 10),      # Rmin (A) -> sigma (nm)
                                                "epsilon": repr(abs(emin) * 4.184)})          # kcal/mol -> kJ/mol
                added.append(f"{other}-{lig}")
    ET.indent(tree)
    out = os.path.join(outdir, os.path.splitext(os.path.basename(stream))[0] + "_cgenff5.xml")
    tree.write(out)
    os.remove(tmp_out)
    print(f"{stream}: RESI {resname} -> template {template_name}, {len(res.atoms)} atoms, "
          f"charge {sum(a.charge for a in res.atoms):+.3f}, ion NBFIX added: {added or 'none'} -> {out}")
