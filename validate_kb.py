import json
import os
import sys

def main():
    kb_dir = "knowledge_base"
    annex_iii_path = os.path.join(kb_dir, "annex_iii.json")
    articles_path = os.path.join(kb_dir, "articles_obligations.json")
    risk_matrix_path = os.path.join(kb_dir, "risk_matrix.json")

    print("==================================================")
    print("EU AI Act Compliance Agent - Knowledge Base Validator")
    print("==================================================")

    # 1. Check file existence and load JSONs
    files_to_check = [annex_iii_path, articles_path, risk_matrix_path]
    loaded_data = {}
    
    for path in files_to_check:
        if not os.path.exists(path):
            print(f"[-] ERROR: File not found: {path}")
            sys.exit(1)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                loaded_data[path] = data
                print(f"[+] Loaded and parsed successfully: {path}")
        except json.JSONDecodeError as e:
            print(f"[-] ERROR: Invalid JSON in {path}: {e}")
            sys.exit(1)

    annex_data = loaded_data[annex_iii_path]
    articles_data = loaded_data[articles_path]
    risk_data = loaded_data[risk_matrix_path]

    # 2. Count and print items in major sections
    categories = annex_data.get("high_risk_categories", [])
    articles = articles_data.get("articles", [])
    risk_levels = risk_data.get("risk_levels", {})
    mappings = risk_data.get("mappings", [])

    print("\n--- Summary Statistics ---")
    print(f"  * High-Risk Categories (Annex III): {len(categories)}")
    print(f"  * Articles with Obligations: {len(articles)}")
    print(f"  * Defined Risk Levels: {len(risk_levels)}")
    print(f"  * Risk Mappings: {len(mappings)}")

    # Collect IDs for cross-reference checking
    category_ids = {cat["id"] for cat in categories if "id" in cat}
    article_ids = {art["id"] for art in articles if "id" in art}

    errors = 0

    print("\n--- Inconsistency Checks ---")

    # 3. Check cross-references in annex_iii.json (articles_ref -> articles)
    for cat in categories:
        cat_id = cat.get("id", "UNKNOWN")
        refs = cat.get("articles_ref", [])
        for ref in refs:
            if ref not in article_ids:
                print(f"[-] ERROR: Category '{cat_id}' references article '{ref}', but it is missing from articles_obligations.json.")
                errors += 1

    # 4. Check cross-references in risk_matrix.json (category_id -> annex_iii, required_articles -> articles)
    for i, mapping in enumerate(mappings):
        m_cat_id = mapping.get("category_id")
        m_req_articles = mapping.get("required_articles", [])
        
        # Check category existence
        if m_cat_id not in category_ids:
            print(f"[-] ERROR: Risk mapping #{i} references category_id '{m_cat_id}', which is missing from annex_iii.json.")
            errors += 1
        
        # Check articles existence
        for ref_art in m_req_articles:
            if ref_art not in article_ids:
                print(f"[-] ERROR: Risk mapping for category '{m_cat_id}' references required article '{ref_art}', which is missing from articles_obligations.json.")
                errors += 1

    # 5. Check if all categories in annex_iii are mapped in risk_matrix
    mapped_categories = {m.get("category_id") for m in mappings if m.get("category_id")}
    for cat_id in category_ids:
        if cat_id not in mapped_categories:
            print(f"[-] WARNING: Category '{cat_id}' has no mapping defined in risk_matrix.json.")

    print("\n--- Validation Results ---")
    if errors == 0:
        print("[+] Success: All cross-references are completely consistent!")
        sys.exit(0)
    else:
        print(f"[-] Failed: Found {errors} cross-reference inconsistencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()
