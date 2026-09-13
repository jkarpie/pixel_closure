#!/usr/bin/env python
"""Regenerate ``DATASETS.md``: what the six full closure suites fit, and why the rest is out.

Every table and count is computed from the sources the suites themselves read, so the file
cannot drift from them silently:

* ``observable_contracts.yaml``, through ``pixel.data.contracts`` -- the selection each suite
  names as ``config.CONTRACT_MODE``, and every record's ``selected_rows`` and ``drop_reason``;
* ``pixel.data.closure_datatypes.DATATYPE_GATES`` -- the dated blocker per observable type;
* PIXEL's ``nnpdf40_dis_dy_roster.yaml`` and ``nnpdf40_dataset_aliases.yaml`` -- NNPDF4.0;
* ``fitpack_legacy/examples/input.py`` -- JAM24, as its uncommented ``xlsx`` entries.

Prose that is measured or argued rather than derived sits between ``<!-- hand:KEY -->`` and
``<!-- /hand:KEY -->`` markers and is carried over verbatim from the file being replaced.
Edit those blocks in ``DATASETS.md`` itself; change anything else here or in the sources.

**A class is never guessed.** For something no selection fits it is, first match wins: the
class word a ``drop_reason`` opens with (``physics (...)``, ``data (...)``,
``closure_builder (...)``); ``duplicate source``; ``all rows fail ... cuts``; the datatype
gate reached through the operator kind or the roster's ``prediction``; otherwise
``unclassified``, printed as such so a human fixes the source.

The two ``_small`` suites are mock smoke tests with no ``CONTRACT_MODE`` and are not listed.

Run with PIXEL's venv (this repo has none of its own)::

    ../pixel/venv/bin/python tools/make_datasets_md.py           # rewrite DATASETS.md
    ../pixel/venv/bin/python tools/make_datasets_md.py --check   # exit 1 with a diff if stale
    ../pixel/venv/bin/python tools/make_datasets_md.py --stdout  # print, write nothing
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from pixel.data.closure_datatypes import DATATYPE_GATES
from pixel.data.contracts import load_observable_contracts, repository_root

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "DATASETS.md"
JAM_INPUT = ROOT.parent / "fitpack_legacy" / "examples" / "input.py"

#: ``(label, contract modes)`` in table order.  The union is one mode per truth package;
#: the two must name the same datasets, and get a row each if they ever stop doing so.
SELECTIONS = (
    ("**union** (JAM u NNPDF)", ("jam_full", "nnpdf_full")),
    ("**JAM24**", ("jam_data",)),
    ("**NNPDF4.0**", ("nnpdf_data",)),
)
JAM_MODE, NNPDF_MODE = "jam_data", "nnpdf_data"

#: Drell-Yan operator kind -> the arm that folds it.  An unknown kind is printed raw.
ARM = {
    "drell_yan": "pointwise",
    "drell_yan.fixed_target.nuclear": "pointwise, nuclear",
    "drell_yan.bin_integrated.fiducial": "fiducial dilepton",
    "drell_yan.mass_spectrum.fiducial": "mass spectrum",
    "drell_yan.bin_integrated.full_phase_space": "full phase space",
}

#: Classes, in the order the tables list them.
CLASS_ORDER = ("fitted", "field", "scope", "cuts", "duplicate", "wiring", "builder", "data",
               "physics", "unclassified")
CLASS_LEGEND = (
    ("scope", "implemented; excluded by an analysis decision", "a decision, not development"),
    ("field", "no field in this fit for it to constrain",
     "adding a sector (FFs, helicity, pion)"),
    ("cuts", "every row falls outside the closure's Q2/W2 cuts", "a change of cuts"),
    ("duplicate", "the same measurement is fitted under another id", "nothing -- counted once"),
    ("wiring", "builder **and** data both present here; nothing reads one into the other",
     "a layout arm"),
    ("builder", "PIXEL has the operator; the closure has no arm for this observable",
     "a closure arm"),
    ("data", "builder exists; numbers or edges are not in the local trees",
     "fetch from HEPData"),
    ("physics", "the coefficient, matching or correction does not exist in PIXEL", "research"),
)

#: A ``drop_reason`` opening with one of these words, then ``(`` or ``:``, states its class.
DROP_PREFIX = {"physics": "physics", "data": "data", "closure_builder": "builder",
               "builder": "builder", "wiring": "wiring", "scope": "scope", "field": "field",
               "cuts": "cuts"}
_DROP_HEAD = re.compile(r"([a-z_]+)\s*[(:]")
#: Otherwise the class is the datatype gate's, reached through the operator kind ...
OPERATOR_GATE = {
    "unimplemented_linear_ratio": "linear_ratio",
    "drell_yan_ratio": "bilinear_ratio",
    "unimplemented heavy-flavour CC dimuon": "sigma_r_cc_dimuon",
    "drell_yan.fiducial_lepton": "drell_yan_fiducial_lepton",
    "drell_yan_total": "drell_yan_total",
}
#: ... or, for a roster entry with no contract at all, through its ``prediction``.
PREDICTION_GATE = {"linear_ratio": "linear_ratio", "bilinear_ratio": "bilinear_ratio"}
GATE_CLASS = {"owner_scope": "scope", "closure_field": "field", "pixel_physics": "physics",
              "closure_builder": "builder"}

#: JAM24 families with no contract record, and the datatype gate that blocks each ...
JAM_FAMILY_GATE = {
    "sia": "sia", "sidis": "sidis_multiplicity", "psidis": "polarized_sidis",
    "pidis": "polarized_dis", "pjet": "polarized_jet", "pp_hadron": "pp_hadron",
    "jet": "inclusive_jet", "ln": "leading_neutron", "wc": "w_plus_charm",
}
#: ... or, where no gate exists, the claim itself with the date it was last checked.
#: Families sharing one claim share a row.
JAM_FAMILY_NOTES = {
    ("pion_qT",): ("field", "Pion-induced Drell-Yan: constrains the pion PDF, which this fit "
                   "does not carry", "2026-09-09"),
    ("SU23", "RpITD", "ccLCS"): ("scope", "JAM's own lattice inputs; the closure injects and "
                                 "recovers its own lattice sector (section 1c)", "2026-09-09"),
    ("wzrv",): ("scope", "every table is an asymmetry or a ratio -- A_eta for W leptons, RW, "
                "and the polarized A_L -- read from each workbook's obs column", "2026-09-10"),
}
#: Single uncontracted tables inside otherwise contracted families.
JAM_TABLE_NOTES = {
    ("dy", "20001"): ("scope", "E866 pd-over-2pp cross-section ratio", "2026-09-09"),
    ("dy", "20002"): ("scope", "E906 pd-over-2pp cross-section ratio", "2026-09-09"),
}

_HERA = ("dis.hera_ep_10026", "dis.hera_ep_10027", "dis.hera_ep_10028", "dis.hera_ep_10029",
         "dis.hera_em_10030", "dis.hera_cc_ep_10031", "dis.hera_cc_em_10032")
#: NNPDF4.0 roster entries the closure fits through their fitpack copy.  The contracts do
#: not record the pairing, so it is stated here -- and checked at every run against the
#: ``inspire_id`` both sides carry, so a wrong pairing fails instead of printing.
NNPDF_FITPACK_TWINS = {
    "NMC": ("dis.nmc_p_10020",),
    "SLACP_dwsh": ("dis.slac_p_10010",),
    "SLACD_dw_ite": ("dis.slac_d_10011",),
    "BCDMSP_dwsh": ("dis.bcdms_p_10016",),
    "BCDMSD_dw_ite": ("dis.bcdms_d_10017",),
    "DYE886P": ("dy.e866_pp_10001",),
    **{name: _HERA for name in ("HERACOMBNCEM", "HERACOMBNCEP460", "HERACOMBNCEP575",
                                "HERACOMBNCEP820", "HERACOMBNCEP920", "HERACOMBCCEM",
                                "HERACOMBCCEP")},
}

#: Hand-written blocks, in the order they appear in the file.
HAND_SLOTS = ("selections_notes", "dis_notes", "dy_intro", "dy_notes", "lattice_intro",
              "differences_notes", "not_included_intro", "short_version", "run_costs")
_HAND = re.compile(r"<!-- hand:([\w-]+) -->\n(.*?)<!-- /hand:\1 -->", re.S)
_INPUT_LINE = re.compile(r"^conf\['datasets'\]\['(\w+)'\]\['xlsx'\]\[([^\]]+)\]\s*=\s*'([^']+)'")
_FITPACK_SOURCE = re.compile(r"fitpack_legacy/database/([^/]+)/expdata/([^/]+)\.xlsx$")


# -- small helpers --------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _short(text, limit: int = 170) -> str:
    """The first sentence of ``text``, cut at ``limit`` characters on a word boundary."""
    text = " ".join(str(text or "").split()).replace("|", "\\|")
    first = re.split(r"(?<=\.)\s", text, maxsplit=1)[0]
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0].rstrip(",;:-") + " ..."
    return first


def _table(columns, rows) -> list[str]:
    """A markdown table; ``columns`` is ``[(title, align)]`` with ``align`` one of ``lrc``."""
    rule = {"l": "---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(title for title, _ in columns) + " |",
           "|" + "|".join(rule[align] for _, align in columns) + "|"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return out


def _mark(flag: bool) -> str:
    return "x" if flag else ""


def _by_class(counter: Counter) -> str:
    items = sorted(counter.items(), key=lambda kv: CLASS_ORDER.index(kv[0]))
    return ", ".join(f"{cls} {n}" for cls, n in items)


def _gate_reason(name: str) -> tuple[str, str]:
    gate = DATATYPE_GATES[name]
    return (GATE_CLASS[gate.status],
            f"{_short(gate.blocker)} (`{name}` gate, measured {gate.blocker_measured})")


def _classify(record=None, prediction=None) -> tuple[str, str]:
    """``(class, reason)`` for something no selection fits -- see the module docstring."""
    text = " ".join(str((record or {}).get("drop_reason") or "").split())
    head = _DROP_HEAD.match(text)
    if head and head.group(1) in DROP_PREFIX:
        return DROP_PREFIX[head.group(1)], _short(text)
    if text.startswith("duplicate source"):
        return "duplicate", _short(text)
    if re.search(r"\ball rows fail\b.*\bcuts\b", text):
        return "cuts", _short(text)
    gate = OPERATOR_GATE.get(str(record["operator"]["kind"])) if record else None
    gate = gate or PREDICTION_GATE.get(str(prediction or ""))
    if gate:
        return _gate_reason(gate)
    return "unclassified", _short(text) or "no drop_reason and no matching datatype gate"


def _section(record_id: str) -> str:
    return "1a" if record_id.startswith("dis.") else "1c" if record_id.startswith("lattice.") \
        else "1b"


# -- the sources ----------------------------------------------------------------------------

def _read_hand_blocks(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    blocks: dict[str, str] = {}
    for match in _HAND.finditer(path.read_text()):
        key, body = match.groups()
        if key in blocks:
            raise SystemExit(f"{path}: hand block {key!r} appears twice")
        blocks[key] = body
    orphans = sorted(set(blocks) - set(HAND_SLOTS))
    if orphans:
        raise SystemExit(
            f"{path}: hand blocks {orphans} have no slot in {Path(__file__).name}. Move their "
            "text into a block that has one, or add a slot, before regenerating -- nothing is "
            "dropped silently."
        )
    return blocks


def _suite_modes() -> dict[str, list[str]]:
    """``CONTRACT_MODE`` -> the suites naming it, read from each config without importing it."""
    modes: dict[str, list[str]] = defaultdict(list)
    for config in sorted(ROOT.glob("closure_*/config.py")):
        for node in ast.parse(config.read_text()).body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and any(isinstance(t, ast.Name) and t.id == "CONTRACT_MODE"
                            for t in node.targets)):
                modes[str(node.value.value)].append(config.parent.name)
    return modes


def _jam_entries() -> list[tuple[str, str, str]]:
    """``(family, table, path)`` for every uncommented ``xlsx`` entry of JAM's ``input.py``."""
    if not JAM_INPUT.exists():
        raise SystemExit(f"JAM24 roster not found: {JAM_INPUT} (needs the sibling "
                         "fitpack_legacy checkout)")
    entries = []
    for line in JAM_INPUT.read_text().splitlines():
        match = _INPUT_LINE.match(line.strip())
        if match:
            family, table, path = match.groups()
            entries.append((family, table.strip("'\""), path))
    return entries


class _Sources:
    def __init__(self):
        self.contracts = load_observable_contracts()
        self.index = {str(r["id"]): r for r in self.contracts["datasets"]}
        self.modes = {str(m): tuple(str(i) for i in spec["dataset_ids"])
                      for m, spec in self.contracts["full_closure_modes"].items()}
        self.suites = _suite_modes()
        self.jam, self.nnpdf = set(self.modes[JAM_MODE]), set(self.modes[NNPDF_MODE])
        self.fitted = set().union(*(self.modes[m] for _, ms in SELECTIONS for m in ms))
        self.pixel_root = repository_root()
        self.roster_doc = yaml.safe_load(
            (self.pixel_root / "nnpdf40_dis_dy_roster.yaml").read_text())
        self.alias_doc = yaml.safe_load(
            (self.pixel_root / "nnpdf40_dataset_aliases.yaml").read_text())
        self.twin_of: dict[str, list[str]] = defaultdict(list)
        for name, ids in NNPDF_FITPACK_TWINS.items():
            for record_id in ids:
                self.twin_of[record_id].append(name)

    def rows(self, ids) -> int:
        return sum(int(self.index[i]["selected_rows"]) for i in ids)

    def source(self, record_id: str) -> str:
        record = self.index[record_id]
        fitpack = _FITPACK_SOURCE.search(str(record["source"]))
        if fitpack:
            text = f"fitpack `{fitpack.group(1)}/{fitpack.group(2)}`"
            names = sorted(self.twin_of.get(record_id, ()))
            if len(names) == 1:
                text += f" (= NNPDF `{names[0]}`)"
            elif names:
                text += f" (= NNPDF `{os.path.commonprefix(names)}*`)"
            return text
        name = record["selector"].get("commondata_name")
        return f"NNPDF `{name}`" if name else _short(record["source"])

    def referenced(self, record_id: str) -> str | None:
        """The other contract a ``duplicate source`` record names, if any."""
        text = str(self.index[record_id]["drop_reason"] or "")
        if not text.startswith("duplicate source"):
            return None
        for token in re.findall(r"[a-z]+\.[a-z0-9_]+", text):
            if token in self.index and token != record_id:
                return token
        return None


# -- the sections ---------------------------------------------------------------------------

def _header(src: _Sources, out: list[str]) -> None:
    contracts_path = src.pixel_root / "observable_contracts.yaml"
    local = ROOT / "observable_contracts.yaml"
    copy = ("this repo's copy is identical" if local.exists()
            and _sha(local) == _sha(contracts_path)
            else "**this repo's copy differs** -- the suites read PIXEL's")
    roster, aliases = src.roster_doc["source"], src.alias_doc["source"]
    out += [
        "# Datasets in the closure fits, and what is left out",
        "",
        "<!-- GENERATED by tools/make_datasets_md.py. Edit the sources, the script, or the "
        "text between hand: markers; everything else is overwritten. -->",
        "",
        f"Generated by `tools/make_datasets_md.py` from `observable_contracts.yaml` (schema "
        f"{src.contracts['schema_version']}, sha256 `{_sha(contracts_path)}`; {copy}), "
        f"`pixel.data.closure_datatypes`, PIXEL's `nnpdf40_dis_dy_roster.yaml` (NNPDF "
        f"{roster['tag']}, `{str(roster['commit'])[:8]}`) with `nnpdf40_dataset_aliases.yaml` "
        f"(`{str(aliases['commit'])[:8]}`), and `fitpack_legacy/examples/input.py` (sha256 "
        f"`{_sha(JAM_INPUT)}`). Regenerate with `../pixel/venv/bin/python "
        "tools/make_datasets_md.py`; `--check` exits 1 when this file is stale.",
        "",
        "Row counts are each contract's `selected_rows` -- **after** the closure's Q2/W2 and DY "
        "cuts, so they are smaller than the published counts. The six full suites are listed; "
        "the two `_small` mock suites are not.",
        "",
    ]


def _selections(src: _Sources, hand: dict, out: list[str]) -> None:
    rows = []
    for label, mode_names in SELECTIONS:
        same = len({frozenset(src.modes[m]) for m in mode_names}) == 1
        for group in ([mode_names] if same else [(m,) for m in mode_names]):
            ids = src.modes[group[0]]
            suites = ", ".join(f"`{s}`" for m in group for s in sorted(src.suites.get(m, ())))
            rows.append((label if same else f"{label} `{group[0]}`", suites or "--",
                         len(ids), src.rows(ids)))
    out += ["## The three selections", ""]
    out += _table([("selection", "l"), ("suites", "l"), ("datasets", "r"), ("rows", "r")], rows)
    out += [
        "",
        "`--truth` and `--data` are independent, so every selection runs against either truth: "
        "`./run_closure.sh --truth NNPDF --data JAM` is the NNPDF4.0 truth on JAM24's datasets.",
        _hand(hand, "selections_notes"),
        "",
        "---",
        "",
    ]


def _included(src: _Sources, hand: dict, out: list[str]) -> None:
    out += ["## 1. Included", "", "### 1a. Deep inelastic scattering", ""]
    dis = sorted(i for i in src.fitted if i.startswith("dis."))
    out += _table(
        [("dataset", "l"), ("source", "l"), ("rows", "r"), ("JAM24", "c"), ("NNPDF4.0", "c")],
        [(f"`{i}`", src.source(i), src.index[i]["selected_rows"], _mark(i in src.jam),
          _mark(i in src.nnpdf)) for i in dis])
    out += ["", f"Subtotal: **{src.rows(dis)} rows** (JAM24 {src.rows(src.jam.intersection(dis))}, "
                f"NNPDF4.0 {src.rows(src.nnpdf.intersection(dis))}).",
            _hand(hand, "dis_notes"), ""]

    out += ["### 1b. Drell-Yan", "", _hand(hand, "dy_intro"), ""]
    order = ("dy", "z", "w")
    dy = sorted((i for i in src.fitted if i.split(".")[0] in order),
                key=lambda i: (order.index(i.split(".")[0]), i))
    out += _table(
        [("dataset", "l"), ("rows", "r"), ("JAM24", "c"), ("NNPDF4.0", "c"), ("arm", "l"),
         ("source", "l")],
        [(f"`{i}`", src.index[i]["selected_rows"], _mark(i in src.jam), _mark(i in src.nnpdf),
          ARM.get(str(src.index[i]["operator"]["kind"]),
                  f"`{src.index[i]['operator']['kind']}`"),
          src.source(i)) for i in dy])
    lepton = [i for i, r in src.index.items()
              if r["operator"]["kind"] == "drell_yan.fiducial_lepton"
              and not str(r["drop_reason"] or "").startswith("duplicate source")]
    out += ["",
            f"Subtotal: **{src.rows(dy)} rows** (JAM24 {src.rows(src.jam.intersection(dy))}, "
            f"NNPDF4.0 {src.rows(src.nnpdf.intersection(dy))}). W lepton-level records "
            f"(`drell_yan.fiducial_lepton`): {len(lepton)} contracted, "
            f"{sum(int(src.index[i]['source_rows']) for i in lepton)} source rows, "
            f"{sum(i in src.fitted for i in lepton)} fitted.",
            "", _hand(hand, "dy_notes"), ""]

    lattice = sorted(i for i in src.fitted if i.startswith("lattice."))
    everywhere = all(set(lattice) <= set(src.modes[m]) for _, ms in SELECTIONS for m in ms)
    out += [f"### 1c. Lattice ({len(lattice)} ensembles, {src.rows(lattice)} rows)", "",
            ("In all three selections. " if everywhere else "Not in every selection. ")
            + _hand(hand, "lattice_intro"), ""]
    half = (len(lattice) + 1) // 2
    pairs = []
    for k in range(half):
        left = lattice[k]
        right = lattice[k + half] if k + half < len(lattice) else None
        pairs.append((f"`{left[len('lattice.'):]}`", src.index[left]["selected_rows"], "",
                      f"`{right[len('lattice.'):]}`" if right else "",
                      src.index[right]["selected_rows"] if right else ""))
    out += _table([("ensemble", "l"), ("rows", "r"), ("", "l"), ("ensemble", "l"),
                   ("rows", "r")], pairs)

    def listing(ids):
        return ", ".join(f"`{i}` ({src.index[i]['selected_rows']})" for i in sorted(ids))

    jam_only, nnpdf_only = src.jam - src.nnpdf, src.nnpdf - src.jam
    both = src.jam & src.nnpdf
    idle = sorted(i for i, r in src.index.items() if r.get("selectable") and i not in src.fitted)
    out += ["", "### 1d. Where the two selections differ", "",
            f"**JAM24 only ({len(jam_only)}):** {listing(jam_only)}.", "",
            f"**NNPDF4.0 only ({len(nnpdf_only)}):** {listing(nnpdf_only)}.", "",
            f"The union is those {len(jam_only) + len(nnpdf_only)} plus the {len(both)} both "
            f"share -- {len(jam_only) + len(nnpdf_only) + len(both)} datasets.", ""]
    if idle:
        out += ["**Selectable but in no selection:** " + "; ".join(
            f"`{i}` ({src.index[i]['selected_rows']} rows) -- "
            f"{_short(src.index[i]['drop_reason']).rstrip('.')}" for i in idle) + ".", ""]
    out += [_hand(hand, "differences_notes"), "", "---", ""]


def _not_included(src: _Sources, hand: dict, out: list[str]) -> None:
    out += ["## 2. Not included, and why", "",
            "Classes, because \"unavailable\" was hiding situations that call for completely "
            "different work. `pixel.data.closure_datatypes` carries these per datatype with a "
            "**dated** blocker; a record's own blocker is its contract's `drop_reason`.", ""]
    out += _table([("class", "l"), ("meaning", "l"), ("what would change it", "l")],
                  [(f"**{c}**", m, w) for c, m, w in CLASS_LEGEND])
    out += ["", _hand(hand, "not_included_intro"), ""]

    # 2a -- JAM24
    by_source = {str(r["source"]): r for r in src.contracts["datasets"]}
    entries = _jam_entries()
    note_group = {family: group for group in JAM_FAMILY_NOTES for family in group}
    groups: dict[tuple, list[str]] = defaultdict(list)
    status = []
    for family, table, path in entries:
        record = by_source.get(f"fitpack_legacy/database/{path}")
        families: tuple = (family,)
        if record is not None and record["id"] in src.jam:
            cls, reason = "fitted", f"section {_section(record['id'])}"
        elif record is not None:
            ref = src.referenced(record["id"])
            if ref and ref in src.jam:
                cls, reason = "fitted", f"as `{ref}` (section {_section(ref)})"
            else:
                cls, reason = _classify(src.index[ref] if ref else record)
        elif (family, table) in JAM_TABLE_NOTES:
            cls, text, date = JAM_TABLE_NOTES[(family, table)]
            reason = f"{text} (checked {date})"
        elif family in JAM_FAMILY_GATE:
            cls, reason = _gate_reason(JAM_FAMILY_GATE[family])
        elif family in note_group:
            families = note_group[family]
            cls, text, date = JAM_FAMILY_NOTES[families]
            reason = f"{text} (checked {date})"
        else:
            cls, reason = "unclassified", "no contract, gate or note for this table"
        groups[(cls, reason, families)].append(table)
        status.append((family, cls))
    per_family = Counter(family for family, _, _ in entries)
    rows = []
    for (cls, reason, families), tables in groups.items():
        label = ", ".join(f"`{f}`" for f in families)
        if len(families) == 1 and len(tables) < per_family[families[0]]:
            label += (" " + ", ".join(tables) if len(tables) <= 3
                      else f" ({len(tables)} of {per_family[families[0]]})")
        rows.append((CLASS_ORDER.index(cls), label, len(tables),
                     len(tables) if cls == "fitted" else 0, cls, reason))
    rows.sort()
    fitted = Counter(family for family, cls in status if cls == "fitted")
    idle = Counter(cls for _, cls in status if cls != "fitted")
    out += [f"### 2a. JAM24 -- {len(entries)} entries in `input.py`, "
            f"{sum(fitted.values())} fitted", ""]
    out += _table([("family", "l"), ("sets", "r"), ("fitted", "r"), ("class", "l"),
                   ("reason", "l")], [row[1:] for row in rows])
    out += ["",
            "Fitted: " + " + ".join(f"`{f}` {n}" for f, n in sorted(fitted.items()))
            + f" = **{sum(fitted.values())}**. Not fitted, {sum(idle.values())}: "
            + _by_class(idle) + ".", ""]

    # 2b -- NNPDF4.0
    roster = src.roster_doc["datasets"]
    aliases = src.alias_doc["aliases"]
    excluded = src.roster_doc["scope"]["excluded_processes"]
    tally: dict[str, list[str]] = defaultdict(list)
    idle_classes: dict[str, Counter] = defaultdict(Counter)
    rows = []
    for entry in roster:
        name = str(entry["legacy_name"])
        ids = _roster_records(entry, aliases, src.contracts["datasets"])
        if not ids and name in NNPDF_FITPACK_TWINS:
            ids = list(NNPDF_FITPACK_TWINS[name])
            for record_id in ids:
                if record_id not in src.index:
                    raise SystemExit(f"NNPDF_FITPACK_TWINS[{name!r}] names unknown {record_id}")
                if entry["inspire_id"] not in src.index[record_id]["inspire_ids"]:
                    raise SystemExit(
                        f"NNPDF_FITPACK_TWINS[{name!r}] -> {record_id}: roster inspire_id "
                        f"{entry['inspire_id']} is not among the contract's "
                        f"{src.index[record_id]['inspire_ids']} -- the pairing is wrong")
        chosen = [i in src.nnpdf for i in ids]
        state = "whole" if ids and all(chosen) else "part" if any(chosen) else "not"
        tally[state].append(name)
        if state == "whole":
            continue
        unselected = [src.index[i] for i in ids if i not in src.nnpdf]
        verdicts = ([_classify(r) for r in unselected] if unselected
                    else [_classify(prediction=entry["prediction"])])
        classes = sorted({c for c, _ in verdicts}, key=CLASS_ORDER.index)
        idle_classes[state].update([classes[0]])
        rows.append((f"`{name}`" + (" (W rows)" if state == "part" else ""),
                     "in part" if state == "part" else "not fitted",
                     sum(int(r["source_rows"]) for r in unselected) if unselected else "--",
                     "/".join(classes), " / ".join(dict.fromkeys(r for _, r in verdicts))))
    out += [f"### 2b. NNPDF4.0 -- {len(roster)} roster entries: {len(tally['whole'])} fitted, "
            f"{len(tally['part'])} in part, {len(tally['not'])} not", "",
            f"{len(excluded)} whole processes are excluded upstream of the roster and are *not* "
            f"among those {len(roster)}: **{', '.join(excluded)}**.", ""]
    out += _table([("dataset", "l"), ("status", "l"), ("rows", "r"), ("class", "l"),
                   ("reason", "l")], rows)
    out += ["",
            f"Fitted whole ({len(tally['whole'])}): "
            + ", ".join(f"`{n}`" for n in tally["whole"]) + ".", "",
            f"Fitted in part ({len(tally['part'])}; the fitted rows are in section 1b): "
            + ", ".join(f"`{n}`" for n in tally["part"]) + ".", "",
            f"Not fitted ({len(tally['not'])}): {_by_class(idle_classes['not'])}.", "",
            "---", ""]
    src.summary = (sum(fitted.values()), len(entries), idle, tally, idle_classes)


def _roster_records(entry, aliases, records) -> list[str]:
    """Contracts for one roster entry by its current commondata name, duplicates dropped."""
    dataset = str(aliases.get(entry["legacy_name"], {}).get("dataset", ""))
    stem = dataset.rsplit("-", 1)[0] if "-" in dataset else dataset
    hits = []
    for record in records:
        name = str(record["selector"].get("commondata_name") or "")
        same = (name == dataset or (stem and name.startswith(stem + "_"))
                or str(record["source"]).rstrip("/").endswith("commondata/" + stem))
        if dataset and same and not str(record["drop_reason"] or "").startswith(
                "duplicate source"):
            hits.append(str(record["id"]))
    return hits


def _summary(src: _Sources, hand: dict, out: list[str]) -> None:
    jam_fitted, jam_total, jam_idle, tally, idle_classes = src.summary
    union = src.modes[SELECTIONS[0][1][0]]
    out += ["## 3. The short version", "",
            f"* **Fitted:** the union's {len(union)} datasets / {src.rows(union)} rows; JAM24 "
            f"{len(src.jam)} / {src.rows(src.jam)}; NNPDF4.0 {len(src.nnpdf)} / "
            f"{src.rows(src.nnpdf)}.",
            f"* **JAM24 `input.py`:** {jam_fitted} of {jam_total} entries fitted; the rest by "
            f"class: {_by_class(jam_idle)}.",
            f"* **NNPDF4.0 roster:** {len(tally['whole'])} fitted whole, {len(tally['part'])} in "
            f"part ({_by_class(idle_classes['part'])}), {len(tally['not'])} not "
            f"({_by_class(idle_classes['not'])}).",
            _hand(hand, "short_version"), "", "---", "",
            "## 4. What running the collider data takes", "", _hand(hand, "run_costs")]


def _hand(blocks: dict[str, str], key: str) -> str:
    return f"<!-- hand:{key} -->\n{blocks.get(key, '')}<!-- /hand:{key} -->"


def render(hand: dict[str, str]) -> str:
    src = _Sources()
    out: list[str] = []
    _header(src, out)
    _selections(src, hand, out)
    _included(src, hand, out)
    _not_included(src, hand, out)
    _summary(src, hand, out)
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 and print a diff if the file is stale")
    parser.add_argument("--stdout", action="store_true",
                        help="print the regenerated file; write nothing")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH,
                        help="the file to regenerate (default: DATASETS.md at the repo root)")
    args = parser.parse_args(argv)
    text = render(_read_hand_blocks(args.path))
    if args.stdout:
        sys.stdout.write(text)
        return 0
    old = args.path.read_text() if args.path.exists() else ""
    if args.check:
        if old == text:
            print(f"{args.path.name} is up to date")
            return 0
        sys.stdout.writelines(difflib.unified_diff(
            old.splitlines(True), text.splitlines(True),
            f"{args.path.name} (on disk)", f"{args.path.name} (regenerated)"))
        return 1
    args.path.write_text(text)
    print(f"wrote {args.path}" + (" (unchanged)" if old == text else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
