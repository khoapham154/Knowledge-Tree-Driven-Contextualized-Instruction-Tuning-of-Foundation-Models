"""
radial_tree.py
Usage:
    python radial_tree.py  # uses the HARD_CODED_TEXT below
    python radial_tree.py --json path/to/Knowledge_MRI_Tree.json  # load JSON
    python radial_tree.py --text path/to/tree.txt                 # load tab text
    python radial_tree.py --max-depth 3                           # prune to max depth
    python radial_tree.py --out output.png --fontsize 8           # customize output
"""

import argparse
import json
import math
import pathlib
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------- 1. Hard-coded tree ------------------------------------------------
HARD_CODED_TEXT = r"""
!Elements
	Developmental-malformation
		Hypothalamic-hamartoma
	Dysplasia
	Early-life-destructive-lesion
		Porencephaly
		Ulegyria
		Hemiatrophy
		Leukomalacia
		Perinatal-ischemic-lesion
	Encephalocele
	Gliosis-or-encephalomalacia-not-otherwise-specified
	Hippocampal-sclerosis
		Abnormal-hippocampal-morphology
			Flattened-or-inclined-hippocampus
		Hippocampal-atrophy
		Hippocampal-hyperintense-T2-or-FLAIR-signal
		Loss-of-normal-internal-hippocampal-structure
	Inflammatory-infectious
		Abscess
		Cysticercosis
		Encephalitis-Other
		Limbic-encephalitis
		Rasmussen-encephalitis
		Sarcoidosis
		Vasculitis
	Lesion
		Lipoma
		Other-cystic-lesion
		Unspecified-lesion
	Malformation-of-cortical-development
		Focal-cortical-dysplasia
			Abnormal-gyrus
			Bottom-of-the-sulcus-dysplasia
			Cortical-thickening
			Poor-grey-white-matter-delineation
		Hemimegalencephaly
		Heterotopia
			Grey-matter-heterotopia
			Periventricular-nodular-heterotopia
			Subcortical-laminar-heterotopia
			Subcortical-nodular-heterotopia
		Lissencephaly-agyria-pachygyria
		Pachymicrogyria
		Polymicrogyria
		Schizencephaly
		Sturge-Weber-syndrome
		Tuberous-sclerosis
			Cortical-hamartomas
	Post-inflammatory-or-infectious-lesion
		Post-abscess
	Post-traumatic-or-post-ischemic-lesion
		Post-operative
		Post-radiation
		Post-stroke
	Stroke-or-haemorrhage
		Arterial-stroke-including-lacunar
		Flow-void-loss
		Haemorrhage-Other
		Intraparenchymal-haemorrhage
		Intraventricular-haemorrhage
		Subarachnoid-haemorrhage
		Subdural-haemorrhage
		Venous-stroke
	Structural-abnormalities
		Chiari-malformation
		Inferior-tonsillar-herniation
	Tumours
		Primary
		Secondary
		Low-grade-tumour
			Aplastic-tumour
			Astrocytoma
				Pleomorphic-astrocytoma
				Diffuse-astrocytoma
			DNET
			Ganglioglioma
			Oligodendroglioma
	Vascular-malformation
		Arteriovenous-malformation
		Cavernous-angioma
		Cerebral-aneurysm
		Developmental-venous-anomaly

!Location-Terms
	Lateralisation
		Left-Lateralisation
		Right-Lateralisation
		Bilateral
	Distribution
		Unifocal
		Multifocal
		Multilobar
		Hemispheric
		Diffuse-or-generalized
	Location
		Cortical
			Frontal
				Dorsolateral-frontal
				Frontal-polar
				Mesial-frontal
				Orbital-frontal
			Parietal
				Dorsolateral-parietal
				Mesial-parietal
			Insula
			Temporal
				Lateral-temporal
				Mesial-temporal
				Temporal-polar
			Occipital
				Basal-occipital
				Lateral-occipital
				Mesial-occipital
		Subcortical
			Basal-ganglia
			Callosum
			Grey-white-matter-junction
			Periventricular
			Thalamus
			Deep-white-matter
			White-matter-other
		Cerebellum
		Brainstem
			Medulla
			Midbrain
			Pons
		Meningeal

Other-Features
	Acute-intracranial-abnormality
	Agenesis
	Arachnoid-cyst
	Atrophy
	Calcification
	Contrast-enhancement
	CSF-space-prominence
	Demyelination
	Diffusion-restriction
	Dysgenesis
	Hydrocephalus
	Hygroma
	Hyperintense-T2-or-FLAIR-signal
	Hypertrophy
	Mass-effect
	Microencephaly
	Oedema
	Small-vessel-ischemic-change
	Thrombus
	Ventriculomegaly
	Virchow-Robinson-space
""".strip("\n")

# ---------- 2. Parsing & Pruning ------------------------------------------------
def parse_tab_text(text: str, indent_char="\t"):  # noqa: C901
    """Turn a tab-indented outline into a nested dict."""
    root = {}
    stack = [(-1, root)]  # (indent level, dict)
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        lvl = 0
        while line.startswith(indent_char):
            lvl += 1
            line = line[len(indent_char):]
        # strip leading '!' markers for headings
        node = line.strip().lstrip("!").strip()
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        parent_dict = stack[-1][1]
        parent_dict.setdefault(node, {})
        stack.append((lvl, parent_dict[node]))
    return root


def prune_tree(nested: dict, max_depth: int, current_depth: int = 0) -> dict:
    """Prune the tree to a maximum depth."""
    if current_depth >= max_depth:
        return {}
    pruned = {}
    for k, v in nested.items():
        pruned[k] = prune_tree(v, max_depth, current_depth + 1)
    return pruned

# ---------- 3. Building Graph --------------------------------------------------
def build_edges(nested, parent=None, edges=None):
    if edges is None:
        edges = defaultdict(list)
    if isinstance(nested, dict):
        for k, v in nested.items():
            if parent is not None:
                edges[parent].append(k)
            build_edges(v, k, edges)
    return edges

# ---------- 4. Radial Layout --------------------------------------------------
def radial_positions(edges, root):
    """Compute (radius, theta) for each node by subtree sizes."""
    sizes = {}
    def subtree(n):
        if n not in edges or not edges[n]:
            sizes[n] = 1
            return 1
        s = 0
        for c in edges[n]:
            s += subtree(c)
        sizes[n] = s
        return s
    subtree(root)
    pos = {}
    def place(n, t0, t1, r):
        theta = (t0 + t1) / 2
        pos[n] = (r, theta)
        if n in edges:
            start = t0
            for c in edges[n]:
                span = (t1 - t0) * sizes[c] / sizes[n]
                place(c, start, start + span, r + 1)
                start += span
    place(root, 0, 2 * math.pi, 0)
    return pos

# ---------- 5. Category Assignment --------------------------------------------
def get_categories(edges, root, color_map):
    """Map each node to one of the top-level categories."""
    parent_map = {}
    for p, children in edges.items():
        for c in children:
            parent_map[c] = p
    def find_cat(n):
        if n == root:
            return None
        p = parent_map.get(n)
        if p == root:
            return n
        return find_cat(p)
    categories = {}
    # include both internal and leaf nodes
    all_nodes = set(edges.keys()) | {c for cs in edges.values() for c in cs}
    for n in all_nodes:
        cat = find_cat(n)
        categories[n] = cat if cat in color_map else None
    return categories

# ---------- 6. Plot -----------------------------------------------------------
def plot_radial(edges, root="MRI Entities", out="tree.png", font_size=6):
    # Define colors for the three main categories
    color_map = {
        "Elements": "C0",
        "Location-Terms": "C3",
        "Other-Features": "C2"
    }
    pos = radial_positions(edges, root)
    categories = get_categories(edges, root, color_map)
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={"polar": True})
    # Draw edges
    for p, children in edges.items():
        for c in children:
            col = color_map.get(categories.get(c), "gray")
            r1, t1 = pos[p]
            r2, t2 = pos[c]
            ax.plot([t1, t2], [r1, r2], color=col, lw=0.6)
    # Draw labels
    for n, (r, t) in pos.items():
        if n == root:
            continue
        ang = math.degrees(t)
        col = color_map.get(categories.get(n), "gray")
        ax.text(t, r, n, fontsize=font_size,
                rotation=ang, rotation_mode="anchor",
                ha="left", va="center", color=col)
    # Center label
    ax.text(0, 0, root, fontsize=font_size * 2, ha="center", va="center")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out, dpi=400, bbox_inches="tight")
    print(f"Saved: {out}")
    svg_out = pathlib.Path(out).with_suffix(".svg")
    fig.savefig(svg_out, bbox_inches="tight")
    print(f"Saved: {svg_out}")

# ---------- 7. CLI ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="Path to JSON file (expects {'entities': {...}})")
    ap.add_argument("--text", help="Path to tab-indented text file")
    ap.add_argument("--out", default="mri_tree.png", help="Output image filename")
    ap.add_argument("--fontsize", type=int, default=6, help="Font size for labels")
    ap.add_argument("--max-depth", type=int, default=3, help="Max depth to display")
    args = ap.parse_args()

    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        nested = data.get("entities", data)
    elif args.text:
        with open(args.text) as f:
            nested = parse_tab_text(f.read())
    else:
        nested = parse_tab_text(HARD_CODED_TEXT)

    # Prune deeply nested branches
    nested = prune_tree(nested, args.max_depth)

    # Build edges and add root
    edges = build_edges(nested)
    root = "MRI Entities"
    edges[root] = list(nested.keys())

    plot_radial(edges, root=root, out=args.out, font_size=args.fontsize)

if __name__ == "__main__":
    main()