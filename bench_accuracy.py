"""Measure retrieval recall, end-to-end pick quality, and per-stage latency."""
import sys, time, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from understand_query import understand_query
from retrieve import find_candidates
from reason import reason_about_query

# (query, acceptable material_ids) - materials a competent engineer would accept
CASES = [
    ("lightweight material for marine use",
     {"al_5052_h32", "al_6061_t6", "ti_cp_grade2", "comp_gfrp_e_glass", "al_3003_o"}),
    ("cheap weldable steel for a bracket",
     {"cs_hsla_a36", "cs_1018_hr", "cs_1018_cd", "cs_1040_hr"}),
    ("knife blade that holds a sharp edge",
     {"ts_d2_hardened", "ts_a2_hardened", "cs_1095_annealed", "ss_410_annealed", "ss_440c"}),
    ("exhaust component that runs at 800 C",
     {"ni_inconel_600", "ss_430_annealed", "ss_303_annealed", "cer_sic", "cer_alumina_96"}),
    ("electrical insulator that survives 1000 C",
     {"cer_alumina_96", "cer_sic"}),
    ("low friction gear that resists fuels",
     {"pl_pom", "pl_nylon66", "pl_ptfe"}),
    ("light and stiff drone frame",
     {"comp_cfrp_ud", "comp_cfrp_qiso", "al_7075_t6", "al_6061_t6", "mg_az31b_f"}),
    ("seawater piping for a boat",
     {"ss_316l_annealed", "ss_316_annealed", "cu_c46400_naval_brass", "ti_cp_grade2",
      "comp_gfrp_e_glass"}),
    ("transparent cover that resists impact",
     {"pl_polycarbonate", "cer_borosilicate_glass"}),
    ("shaft carrying 600 MPa",
     {"as_4340_qt540", "as_4140_qt540", "as_4130_qt540", "ss_17_4ph_h1025",
      "ti_6al_4v_annealed", "ti_6al_4v_solution_aged", "cs_1050_qt425"}),
]

POOL_SIZE = 18
rows = []
t_understand = t_retrieve = t_total = 0.0

for query, good in CASES:
    t0 = time.time()
    u = understand_query(query) or {}
    t1 = time.time()
    sem = u.get("semantic_query", query)
    pool = find_candidates(sem, filters=u.get("filters") or None,
                           family=(u.get("extracted_constraints") or {}).get("material_family"),
                           pool_size=POOL_SIZE)
    t2 = time.time()

    pool_ids = [m["material_id"] for m in pool]
    recall = bool(good & set(pool_ids))
    rank = next((i + 1 for i, m in enumerate(pool_ids) if m in good), None)

    out = reason_about_query(query, top_k=3)
    t3 = time.time()
    picks = [m["material_id"] for m in out.get("materials", [])]
    top1_ok = bool(picks and picks[0] in good)
    any_ok = bool(set(picks) & good)

    t_understand += t1 - t0
    t_retrieve += t2 - t1
    t_total += t3 - t0

    rows.append({
        "query": query, "in_pool": recall, "best_rank_in_pool": rank,
        "top1_ok": top1_ok, "any_ok": any_ok,
        "picks": [m.get("common_name") for m in out.get("materials", [])],
        "pool_head": [m["common_name"] for m in pool[:5]],
        "secs": round(t3 - t0, 1),
    })
    print(f"  done: {query[:44]:<46} pool={'Y' if recall else 'N'} "
          f"top1={'Y' if top1_ok else 'N'}  {round(t3-t0,1)}s")

n = len(CASES)
print("\n" + "=" * 74)
print("RESULTS")
print("=" * 74)
for r in rows:
    flag = "OK  " if r["top1_ok"] else ("weak" if r["any_ok"] else "MISS")
    print(f"\n[{flag}] {r['query']}")
    print(f"        in pool: {r['in_pool']} (best acceptable at pool rank {r['best_rank_in_pool']})")
    print(f"        picked : {r['picks']}")
    if not r["top1_ok"]:
        print(f"        pool[:5]: {r['pool_head']}")

print("\n" + "=" * 74)
print(f"Retrieval recall (acceptable answer in pool): {sum(r['in_pool'] for r in rows)}/{n}")
print(f"Top-1 correct:                                {sum(r['top1_ok'] for r in rows)}/{n}")
print(f"Any correct in top-3:                         {sum(r['any_ok'] for r in rows)}/{n}")
print(f"\nLatency per query: understand {t_understand/n:.1f}s | retrieve {t_retrieve/n:.1f}s "
      f"| TOTAL {t_total/n:.1f}s")
print("=" * 74)
